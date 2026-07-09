from typing import Optional, Tuple
import numpy as np
from numpy import ndarray
from .form import Form, FormExtraParams
from ..basis import AbstractBasis
class LinearForm(Form):
    """A linear form for finite element assembly.
    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take two parameters ``v`` and ``w``.
    """
    def _assemble(self,
                  ubasis: AbstractBasis,
                  vbasis: Optional[AbstractBasis] = None,
                  **kwargs) -> Tuple[ndarray,
                                     ndarray,
                                     Tuple[int],
                                     Tuple[int]]:
        assert vbasis is None
        vbasis = ubasis
        nt = vbasis.nelems
        dx = vbasis.dx
        w = FormExtraParams({
            **vbasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })
        # initialize COO data structures
        sz = vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int32)
        for i in range(vbasis.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i]
            data[ixs] = self._kernel(vbasis.basis[i], w, dx)
        return np.array([rows]), data, (vbasis.N,), (vbasis.Nbfun,)

    def _kernel(self, v, w, dx):
            a = self.form(*v, w)
            if getattr(a, "ndim", None) == 2:
                if type(a) is np.ndarray and type(dx) is np.ndarray:
                    a_shape = a.shape
                    dx_shape = dx.shape
                    if dx_shape == a_shape:
                        if np.result_type(a.dtype, dx.dtype).kind in "fc":
                            return np.einsum("ij,ij->i", a, dx)
                    elif dx.ndim == 1 and dx_shape[0] == a_shape[1]:
                        if np.result_type(a.dtype, dx.dtype).kind in "fc":
                            return np.einsum("ij,j->i", a, dx)
                flags = getattr(a, "flags", None)
                if (
                    flags is not None
                    and flags.writeable
                    and flags.owndata
                    and np.result_type(a, dx) == a.dtype
                ):
                    np.multiply(a, dx, out=a)
                    return a.sum(axis=1)
            return np.sum(a * dx, axis=1)
