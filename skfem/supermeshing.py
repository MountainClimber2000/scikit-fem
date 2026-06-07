import numpy as np
from skfem.mesh import MeshLine, MeshTri
from skfem.quadrature import get_quadrature

def intersect(m1, m2):
    """Create a supermesh between two nonmatching (1D or 2D) meshes.

    Parameters
    ----------
    m1
        The first mesh.
    m2
        The second mesh.

    Returns
    -------
    supermesh
        Mesh object with line (1D) or triangle (2D) elements.
    ix1
        Index of each supermesh element within the first mesh.
    ix2
        Index of each supermesh element within the second mesh.

    """
    p1, t1 = m1
    p2, t2 = m2
    if p1.shape[0] == 1 and p2.shape[0] == 1:
        return _intersect1d(p1, t1, p2, t2)
    elif p1.shape[0] == 2 and p2.shape[0] == 2:
        return _intersect2d(p1, t1, p2, t2)
    raise NotImplementedError("The given mesh types not supported.")

def _intersect2d(p1, t1, p2, t2):
    """Two-dimensional supermesh using shapely and bruteforce."""
    try:
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        from shapely.ops import triangulate
    except Exception:
        raise Exception("2D supermeshing requires the package 'shapely>=2'.")

    polys = [Polygon(p1[:, t1[:, itr]].T) for itr in range(t1.shape[1])]
    tree = STRtree(polys)
    geometries = tree.geometries
    query = tree.query
    triangulate_ = triangulate

    p_blocks = []
    ix1 = []
    ix2 = []
    append_p = p_blocks.append
    append_ix1 = ix1.append
    append_ix2 = ix2.append

    for jtr in range(t2.shape[1]):
        poly1 = Polygon(p2[:, t2[:, jtr]].T)
        result = query(poly1)
        if len(result) == 0:
            continue
        for itr in result:
            intersection = poly1.intersection(geometries[itr])
            if intersection.is_empty:
                continue
            for tri in triangulate_(intersection):
                append_p(np.asarray(tri.exterior.xy)[:, :-1])
                append_ix1(itr)
                append_ix2(jtr)

    ntris = len(p_blocks)
    if ntris:
        p = np.hstack(p_blocks)
        t = np.arange(3 * ntris, dtype=np.float64).reshape(ntris, 3).T
    else:
        p = np.empty((2, 0))
        t = np.empty((3, 0))

    return (
        MeshTri(p, t),
        np.array(ix1, dtype=np.int32),
        np.array(ix2, dtype=np.int32),
    )
def _intersect1d(p1, t1, p2, t2):
    """One-dimensional supermesh."""
    # Find unique supermesh facets by combining nodes from both
    # sides in the intersection of x-ranges.
    p1f = p1.flatten().round(decimals=10)
    p2f = p2.flatten().round(decimals=10)

    xmin = max(p1f.min(), p2f.min())
    xmax = min(p1f.max(), p2f.max())

    p1f = p1f[(p1f >= xmin) & (p1f <= xmax)]
    p2f = p2f[(p2f >= xmin) & (p2f <= xmax)]
    p = np.concatenate((p1f, p2f))
    p = np.unique(p)
    t = np.array([np.arange(len(p) - 1), np.arange(1, len(p))])
    p = np.array([p])

    supermap = MeshLine(p, t)._mapping()
    mps = supermap.F(np.array([[.5]]))
    ix1 = MeshLine(p1, t1).element_finder()(mps[0, :, 0])
    ix2 = MeshLine(p2, t2).element_finder()(mps[0, :, 0])

    return MeshLine(p, t), ix1, ix2

def elementwise_quadrature(mesh, supermesh=None, tind=None, intorder=None):
    """For creating element-by-element quadrature rules.

    Parameters
    ----------
    mesh
        The mesh for which to create the quadrature rules.
    supermesh
        Created using skfem.supermeshing.intersect.
    tind
        A subset of elements
    intorder
        Integration order, by default equal to 4.

    """
    if intorder is None:
        intorder = 4
    if supermesh is None:
        raise Exception("elementwise_quadrature: User must provide "
                        "'supermesh' keyword argument which has been "
                        "created using skfem.supermeshing.intersect.")
    X, W = get_quadrature(supermesh.elem, intorder)
    mmap = mesh.mapping()
    smap = supermesh.mapping()
    return (
        mmap.invF(smap.F(X), tind=tind),
        np.abs(smap.detDF(X) / mmap.detDF(X, tind=tind)) * W,
    )
