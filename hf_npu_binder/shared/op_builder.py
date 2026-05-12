"""JIT C++ extension builder for AscendC / aclnn ops.

Pattern (vendored from MindSpeed's ``MindSpeedOpBuilder`` with simplifications):

  1. Subclass ``AscendcOpBuilder`` and override ``sources()`` to list the
     .cpp files for this op (relative to the binder package root).
  2. First call to ``load()`` triggers ``torch.utils.cpp_extension.load(...)``
     which JIT-compiles the cpp into a .so against the user's torch /
     torch_npu / CANN install. The compiled .so is cached on disk under
     ``~/.cache/torch_extensions/<py_ver>_<torch_ver>/<op_name>/`` so
     subsequent runs skip the ~10s compile.
  3. The PYBIND11_MODULE in each .cpp defines the Python entry points
     for the op. ``load()`` returns the loaded module; callers do
     ``op.metadata(...)`` / ``op.forward(...)`` / etc.

Cross-process caching is handled by torch_extensions/. Within-process
caching is the ``_loaded_ops`` class dict — subsequent ``load()`` calls on
the same op name return the cached module reference without going
through ``torch.utils.cpp_extension.load`` again.

Why JIT instead of pre-compiled wheels:
  - Multiple CANN versions / torch_npu versions / Python versions /
    aarch64+x86 cross-product = a lot of wheels to ship and keep in
    sync. JIT against the user's exact install sidesteps that.
  - User pays ~10s one-time compile vs us shipping & maintaining wheels
    for every (torch_npu, CANN, py_ver, arch) combination.

Trade-offs:
  - First call ~10s. Subsequent calls fast (cached .so).
  - User machine needs gcc + complete dev headers. NPU dev boxes always
    have these (CANN install provides them); this is a CPU-dev-box
    blocker we accept (kernel is NPU-only anyway).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import torch
from torch.utils.cpp_extension import load


ASCEND_HOME_PATH = "ASCEND_HOME_PATH"


class AscendcOpBuilder(ABC):
    """Base class for AscendC / aclnn JIT C++ extensions.

    Subclasses declare which .cpp file(s) to compile via ``sources()``.
    The base class handles include paths, ldflags and caching, so the
    subclass implementation is a 3-line override::

        class FooOpBuilder(AscendcOpBuilder):
            OP_NAME = "foo"

            def sources(self):
                return ["deepseek_v4/csrc/foo.cpp"]

    Default include search path includes:
      * ``$ASCEND_HOME_PATH/include``      — CANN headers (acl/, aclnn/, ...)
      * ``<torch_npu>/include``            — torch_npu C++ API
      * ``<torch_npu>/include/third_party/{acl,hccl}/inc`` — adapter shims
      * ``hf_npu_binder/shared/csrc/inc``  — our vendored ACLNN dispatcher

    Default ldflags link ``-lascendcl`` (CANN runtime) and ``-ltorch_npu``
    (PyTorch's NPU backend lib).

    Per-process caching keeps the loaded module reference in
    ``_loaded_ops`` so multiple calls to ``load()`` for the same op name
    only do ``dlopen`` once.
    """

    OP_NAME: str = ""  # subclass overrides

    _loaded_ops: dict = {}  # class-level cache: op_name -> loaded module
    _cann_path: str | None = None
    _torch_npu_path: str | None = None

    def __init__(self, name: str | None = None):
        self.name = name or self.OP_NAME
        if not self.name:
            raise ValueError(
                f"{type(self).__name__}: OP_NAME class attribute must be set "
                f"or `name` kwarg passed to __init__"
            )
        # CANN / torch_npu lookups are DEFERRED to load(). __init__ stays
        # cheap so module import works on CPU dev boxes (without sourced
        # CANN toolkit / torch_npu install). Subclasses can do
        # ``Cls()`` at module scope without forcing the dependency.
        self._cann_path = None
        self._torch_npu_path = None

    def _resolve_env(self) -> None:
        """Populate ``_cann_path`` and ``_torch_npu_path`` from the running
        environment. Idempotent — re-entering after a successful resolve
        is a no-op. Called at the top of :meth:`load`; raises clearly if
        CANN or torch_npu can't be located."""
        if self._cann_path is not None and self._torch_npu_path is not None:
            return
        cann_path = self._get_cann_path()
        if cann_path is None:
            raise RuntimeError(
                f"Cannot locate CANN install: env var {ASCEND_HOME_PATH} is "
                f"unset or points to a non-existent path. Source the CANN "
                f"toolkit set_env.sh before calling .load() on this op."
            )
        try:
            import torch_npu
        except ImportError as e:
            raise RuntimeError(
                f"torch_npu is required to build ascendc ops; got ImportError: {e}"
            ) from e
        self._cann_path = cann_path
        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))

    # ---- subclass hook -------------------------------------------------
    @abstractmethod
    def sources(self) -> List[str]:
        """Return list of .cpp source paths *relative to the binder package
        root* (i.e. relative to ``hf_npu_binder/`` directory). The base
        ``load()`` resolves them to absolute paths.
        """
        ...

    # ---- path / arg helpers (override-friendly) ------------------------
    def include_paths(self) -> List[str]:
        """Include search paths for the compile. Subclass can extend if
        a particular op needs additional headers (rare)."""
        binder_pkg_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        return [
            os.path.join(self._cann_path, "include"),
            os.path.join(self._torch_npu_path, "include"),
            os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
            os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
            # Our vendored ACLNN dispatcher — bare ``#include "aclnn_common.h"``
            # resolves through this dir.
            os.path.join(binder_pkg_root, "shared/csrc/inc"),
        ]

    def cxx_args(self) -> List[str]:
        # Same hardening flags as MindSpeed uses, plus -O2.
        return [
            "-fstack-protector-all",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-fPIC",
            "-pie",
            "-s",
            "-fvisibility=hidden",
            "-D_FORTIFY_SOURCE=2",
            "-O2",
        ]

    def extra_ldflags(self) -> List[str]:
        return [
            "-L" + os.path.join(self._cann_path, "lib64"),
            "-lascendcl",
            "-L" + os.path.join(self._torch_npu_path, "lib"),
            "-ltorch_npu",
        ]

    # ---- internals -----------------------------------------------------
    def _get_cann_path(self) -> str | None:
        path = os.environ.get(ASCEND_HOME_PATH)
        if path and os.path.exists(path):
            return path
        return None

    def _abs_sources(self) -> List[str]:
        binder_pkg_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        return [os.path.join(binder_pkg_root, src) for src in self.sources()]

    # ---- public entry --------------------------------------------------
    def load(self, verbose: bool = True):
        """Resolve the C++ module, JIT-compiling on first call.

        Returns the loaded module (the pybind-exposed namespace). Repeated
        calls on the same op name return the cached module reference; the
        underlying .so is cached on disk by ``torch_extensions`` and only
        recompiled if the source / build flags change.
        """
        if self.name in AscendcOpBuilder._loaded_ops:
            return AscendcOpBuilder._loaded_ops[self.name]

        # First .load(): now we actually need CANN + torch_npu to be present.
        self._resolve_env()

        module = load(
            name=self.name,
            sources=self._abs_sources(),
            extra_include_paths=self.include_paths(),
            extra_cflags=self.cxx_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
        )
        AscendcOpBuilder._loaded_ops[self.name] = module
        return module


__all__ = ["AscendcOpBuilder", "ASCEND_HOME_PATH"]
