"""One-electron integrals involving Contracted Cartesian Gaussians."""
import numpy as np
from scipy.special import hyp1f1


def _compute_one_elec_integrals(
    coord_point,
    coord_a,
    angmom_a,
    angmoms_a,
    exps_a,
    coeffs_a,
    norm_a,
    coord_b,
    angmom_b,
    angmoms_b,
    exps_b,
    coeffs_b,
    norm_b,
):
    """Return the one-electron integrals for a point charge interaction.

    Parameters
    ----------
    coord_point : np.ndarray(3,)
            Center of the point charge?
    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmoms_a : np.ndarray(L_a, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the left side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coeffs_a : np.ndarray(K_a, M_a)
        Contraction coefficients of the primitives on the left side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_a : np.ndarray(L_a, K_a)
        Normalization constants for the primitives in each contraction on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmoms_b : np.ndarray(L_b, 3)
        Angular momentum vectors (lx, ly, lz) for the contractions on the right side.
        Note that a two dimensional array must be given, even if there is only one angular momentum
        vector.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.
    coeffs_b : np.ndarray(K_b, M_b)
        Contraction coefficients of the primitives on the right side.
        The coefficients always correspond to generalized contractions, i.e. two-dimensional array
        where the first index corresponds to the primitive and the second index corresponds to the
        contraction (with the same exponents and angular momentum).
    norm_b : np.ndarray(L_b, K_b)
        Normalization constants for the primitives in each contraction on the right side.

    # FIXME: finish docstring
    Returns
    -------
    integrals :

    """
    # TODO: Overlap screening

    # NOTE: following convention will be used to organize the axis of the multidimensional arrays
    # axis 0 : l (size: l_a + l_b + 1)
    # axis 1 : a_x (size: l_a + l_b + 1)
    # axis 2 : a_y (size: l_a + l_b + 1)
    # axis 3 : a_z (size: l_a + l_b + 1)
    # axis 4 : b_x (size: l_a + l_b + 1)
    # axis 5 : b_y (size: l_a + l_b + 1)
    # axis 6 : b_z (size: l_a + l_b + 1)
    # axis 7 : dimension (x,y,z) of coordinate (size: 3)
    # axis 8 : primitive of contraction b (size: K_b)
    # axis 9 : primitive of contraction a (size: K_a)

    integrals = np.zeros(
        (angmom_a + angmom_b + 1,
         angmom_a + angmom_b + 1, angmom_a + angmom_b + 1, angmom_a + angmom_b + 1,
         angmom_a + angmom_b + 1, angmom_a + angmom_b + 1, angmom_a + angmom_b + 1,
         3, exps_b.size, exps_a.size)
    )

    # Adjust axes
    coord_point = coord_point[np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis,np.newaxis,np.newaxis, :, np.newaxis, np.newaxis]
    coord_a = coord_a[np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis,np.newaxis,np.newaxis, :, np.newaxis, np.newaxis]
    coord_b = coord_b[np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis,np.newaxis,np.newaxis, :, np.newaxis, np.newaxis]
    exps_a = exps_a[np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis, np.newaxis, :]
    exps_b = exps_b[np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis,np.newaxis,np.newaxis, np.newaxis, :, np.newaxis]

    # sum of the exponents
    exps_sum = exps_a + exps_b
    # coordinate of the weighted average center
    coord_wac = (exps_a * coord_a + exps_b * coord_b) / exps_sum
    # relative distance from weighted average center
    rel_coord_a = coord_wac - coord_a           # R_pa
    rel_coord_b = coord_wac - coord_b           # R_pb
    rel_dist = coord_a - coord_b                # R_ab
    rel_coord_point = coord_wac - coord_point   # R_pc
    # harmonic mean
    harm_mean = exps_a * exps_b / exps_sum

    # Initialize V(m)(000|000) for all m
    # NOTE: There's some documented instability for hyp1f1, mainly for large values or complex numbers.
    # In this case it seems fine, but could be a source of unwanted behaviour.
    for m in range(angmom_a + angmom_b + 1):
        integrals[m, 0:1,0:1,0:1, 0:1,0:1,0:1, :, :, :] = hyp1f1(
            m + 1/2, m + 3/2, (-exps_sum * np.abs(rel_coord_point) ** 2)
        ) / (2 * m + 1) * np.exp(-harm_mean * np.abs(coord_a - coord_b) ** 2)

    # Vertical recursion for one nonzero index i.e. V(010|000)

    return integrals





