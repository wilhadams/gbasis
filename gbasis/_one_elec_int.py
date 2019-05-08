"""One-electron integrals involving Contracted Cartesian Gaussians."""
import numpy as np
from scipy.special import hyp1f1


def _compute_one_elec_integrals(
    coord_point,
    boys_func,
    coord_a,
    angmom_a,
    exps_a,
    coeffs_a,
    coord_b,
    angmom_b,
    exps_b,
    coeffs_b,
):
    """Return the one-electron integrals for a point charge interaction.

    Parameters
    ----------
    coord_point : np.ndarray(3,)
        Center of the point charge.
    boys_func : function

    coord_a : np.ndarray(3,)
        Center of the contraction on the left side.
    angmom_a : int
        Angular momentum of the contraction on the left side.
    exps_a : np.ndarray(K_a,)
        Values of the (square root of the) precisions of the primitives on the left side.
    coord_b : np.ndarray(3,)
        Center of the contraction on the right side.
    angmom_b : int
        Angular momentum of the contraction on the right side.
    exps_b : np.ndarray(K_b,)
        Values of the (square root of the) precisions of the primitives on the right side.

    # FIXME: finish docstring
    Returns
    -------
    integrals : np.ndarray(L_a + L_b + 1, L_a + L_b + 1, L_a + L_b + 1, L_b + 1, L_b + 1, L_b + 1)


    """
    
    m_max = angmom_a + angmom_b + 1

    # NOTE: Ordering convention for vertical recursion of integrals
    # axis 0 : m (size: m_max)
    # axis 1 : a_x (size: m_max)
    # axis 2 : a_y (size: m_max)
    # axis 3 : a_z (size: m_max)
    # axis 4 : primitive of contraction b (size: K_b)
    # axis 5 : primitive of contraction a (size: K_a)

    integrals = np.zeros((m_max, m_max,m_max,m_max, exps_b.size, exps_a.size))

    # Adjust axes for pre-work
    # axis 0 : primitive of contraction b (size: K_b)
    # axis 1 : primitive of contraction a (size: K_a)
    # axis 2 : components of vectors (x, y, z) (size: 3)
    coord_point = coord_point[np.newaxis, np.newaxis, :]
    coord_a = coord_a[np.newaxis, np.newaxis, :]
    coord_b = coord_b[np.newaxis, np.newaxis, :]
    exps_a = exps_a[np.newaxis, :, np.newaxis]
    exps_b = exps_b[:, np.newaxis, np.newaxis]
    coeffs_a = coeffs_a[np.newaxis, :]
    coeffs_b = coeffs_b[:, np.newaxis]
    coeffs = (coeffs_a * coeffs_b).squeeze(axis=-1)


    # sum of the exponents
    exps_sum = exps_a + exps_b
    # coordinate of the weighted average center
    coord_wac = (exps_a * coord_a + exps_b * coord_b) / exps_sum
    # relative distance from weighted average center
    rel_coord_a = coord_wac - coord_a           # R_pa
    rel_dist = coord_a - coord_b                # R_ab
    rel_coord_point = coord_wac - coord_point   # R_pc
    # harmonic mean
    harm_mean = exps_a * exps_b / exps_sum

    # Initialize V(m)(000|000) for all m
    for m in range(m_max):
        integrals[m, 0:1,0:1,0:1, :, :] = (
            boys_func(m, exps_sum.squeeze() * (rel_coord_point ** 2).sum(2))
            * np.exp(-harm_mean.squeeze() * (rel_dist ** 2).sum(2))
        )

    # Vertical recursion for one nonzero index i.e. V(010|000)
    # Slice to avoid if statement
    # For a = 0:
    # Increment a_x:
    integrals[:-1, 1,0,0, :, :] = \
        rel_coord_a[:, :, 0]*integrals[:-1, 0,0,0, :, :] \
        - rel_coord_point[:, :, 0]*integrals[1:, 0,0,0, :, :]
    # Increment a_y:
    integrals[:-1, 0,1,0, :, :] = \
        rel_coord_a[:, :, 1] * integrals[:-1, 0,0,0, :, :] \
        - rel_coord_point[:, :, 1] * integrals[1:, 0,0,0, :, :]
    # Increment a_z
    integrals[:-1, 0,0,1, :, :] = \
        rel_coord_a[:, :, 2] * integrals[:-1, 0,0,0, :, :] \
        - rel_coord_point[:, :, 2] * integrals[1:, 0,0,0, :, :]
    # For a > 0:
    for a in range(1, m_max - 1):
        # Increment a_x:
        integrals[:-a-1, a+1,0,0, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a,0,0, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a,0,0, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                    integrals[:-a-1, a-1,0,0, :, :] - integrals[1:-a, a-1,0,0, :, :]
            )
        # Increment a_y:
        integrals[:-a-1, 0,a+1,0, :, :] = \
            rel_coord_a[:, :, 1] * integrals[:-a-1, 0,a,0, :, :] \
            - rel_coord_point[:, :, 1] * integrals[1:-a, 0,a,0, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                    integrals[:-a-1, 0,a-1,0, :, :] - integrals[1:-a, 0,a-1,0, :, :]
            )
        # Increment a_z
        integrals[:-a-1, 0,0,a+1, :, :] = \
            rel_coord_a[:, :, 2] * integrals[:-a-1, 0,0,a, :, :] \
            - rel_coord_point[:, :, 2] * integrals[1:-a, 0,0,a, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                    integrals[:-a-1, 0,0,a-1, :, :] - integrals[1:-a, a-1,0,0, :, :]
            )

    # Vertical recursion for two nonzero indices i.e. V(110|000)
    # Slice to avoid if statement
    # For a = 0:
    # Increment a_x for all a_y:
    integrals[:-1, 1,1:-1,0, :, :] = \
        rel_coord_a[:, :, 0]*integrals[:-1, 0,1:-1,0, :, :] \
        - rel_coord_point[:, :, 0]*integrals[1:, 0,1:-1,0, :, :]
    # Increment a_x for all a_z:
    integrals[:-1, 1,0,1:-1, :, :] = \
        rel_coord_a[:, :, 0]*integrals[:-1, 0,0,1:-1, :, :] \
        - rel_coord_point[:, :, 0]*integrals[1:, 0,0,1:-1, :, :]
    # Increment a_y for all a_x:
    integrals[:-1, 1:-1,1,0, :, :] = \
        rel_coord_a[:, :, 1] * integrals[:-1, 1:-1,0,0, :, :] \
        - rel_coord_point[:, :, 1] * integrals[1:, 1:-1,0,0, :, :]
    # Increment a_y for all a_z
    integrals[:-1, 0,1,1:-1, :, :] = \
        rel_coord_a[:, :, 1] * integrals[:-1, 0,0,1:-1, :, :] \
        - rel_coord_point[:, :, 1] * integrals[1:, 0,0,1:-1, :, :]
    # Increment a_z for all a_x
    integrals[:-1, 1:-1,0,1, :, :] = \
        rel_coord_a[:, :, 2] * integrals[:-1, 1:-1,0,0, :, :] \
        - rel_coord_point[:, :, 2] * integrals[1:, 1:-1,0,0, :, :]
    # Increment a_z for all a_y
    integrals[:-1, 0,1:-1,1, :, :] = \
        rel_coord_a[:, :, 2] * integrals[:-1, 0,1:-1,0, :, :] \
        - rel_coord_point[:, :, 2] * integrals[1:, 0,1:-1,0, :, :]
    # For a > 0:
    for a in range(1, m_max-1):
        # Increment a_x for all a_y:
        integrals[:-a-1, a+1,a+1:-a-1,0, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a,a+1:-a-1,0, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a,a+1:-a-1,0, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a-1,a+1:-a-1,0, :, :] - integrals[1:-a, a-1,a+1:-a-1,0, :, :]
        )
        # Increment a_x for all a_z:
        integrals[:-a-1, a+1,0,a+1:-a-1, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a,0,a+1:-a-1, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a,0,a+1:-a-1, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a-1,0,a+1:-a-1, :, :] - integrals[1:-a, a-1,0,a+1:-a-1, :, :]
        )
        # Increment a_y for all a_x:
        integrals[:-a-1, a+1:-a-1,a+1,0, :, :] = \
            rel_coord_a[:, :, 1]*integrals[:-a-1, a+1:-a-1,a,0, :, :] \
            - rel_coord_point[:, :, 1]*integrals[1:-a, a+1:-a-1,a,0, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a+1:-a-1,a-1,0, :, :] - integrals[1:-a, a+1:-a-1,a-1,0, :, :]
        )
        # Increment a_y for all a_z
        integrals[:-a-1, 0,a+1,a+1:-a-1, :, :] = \
            rel_coord_a[:, :, 1]*integrals[:-a-1, 0,a,a+1:-a-1, :, :] \
            - rel_coord_point[:, :, 1]*integrals[1:-a, 0,a,a+1:-a-1, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, 0,a-1,a+1:-a-1, :, :] - integrals[1:-a, 0,a-1,a+1:-a-1, :, :]
        )
        # Increment a_z for all a_x
        integrals[:-a-1, a+1:-a-1,0,a+1, :, :] = \
            rel_coord_a[:, :, 2]*integrals[:-a-1, a+1:-a-1,0,a, :, :] \
            - rel_coord_point[:, :, 2]*integrals[1:-a, a+1:-a-1,0,a, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a+1:-a-1,0,a-1, :, :] - integrals[1:-a, a+1:-a-1,0,a-1, :, :]
        )
        # Increment a_z for all a_y
        integrals[:-a-1, 0,a+1:-a-1,a+1, :, :] = \
            rel_coord_a[:, :, 2]*integrals[:-a-1, 0,a+1:-a-1,a, :, :] \
            - rel_coord_point[:, :, 2]*integrals[1:-a, 0,a+1:-a-1,a, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, 0,a+1:-a-1,a-1, :, :] - integrals[1:-a, 0,a+1:-a-1,a-1, :, :]
        )

    # Vertical recursion for three nonzero indices i.e. V(111|000)
    # Slice to avoid if statement
    # For a = 0:
    integrals[:-2, 1,1:-1,1:-1, :, :] = \
        rel_coord_a[:, :, 0]*integrals[:-2, 0,1:-1,1:-1, :, :] \
        - rel_coord_point[:, :, 0]*integrals[1:-1, 0,1:-1,1:-1, :, :]
    integrals[:-2, 1:-1,1,1:-1, :, :] = \
        rel_coord_a[:, :, 1]*integrals[:-2, 1:-1,0,1:-1, :, :] \
        - rel_coord_point[:, :, 1]*integrals[1:-1, 1:-1,0,1:-1, :, :]
    integrals[:-2, 1:-1,1:-1,1, :, :] = \
        rel_coord_a[:, :, 2]*integrals[:-2, 1:-1,1:-1,0, :, :] \
        - rel_coord_point[:, :, 2]*integrals[1:-1, 1:-1,1:-1,0, :, :]
    # For a > 0:
    for a in range(1, m_max-1):
        # Increment a_x for all a_y, a_z:
        integrals[:-a-1, a+1,a+1:-a-1,a+1:-a-1, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a,a+1:-a-1,a+1:-a-1, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a,a+1:-a-1,a+1:-a-1, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a-1,a+1:-a-1,a+1:-a-1, :, :]
                - integrals[1:-a, a-1,a+1:-a-1,a+1:-a-1, :, :]
        )
        # Increment a_y for all a_x, a_z:
        integrals[:-a-1, a+1:-a-1,a+1,a+1:-a-1, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a+1:-a-1,a,a+1:-a-1, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a+1:-a-1,a,a+1:-a-1, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a+1:-a-1,a-1,a+1:-a-1, :, :]
                - integrals[1:-a, a+1:-a-1,a-1,a+1:-a-1, :, :]
        )
        # Increment a_z for all a_x, a_y:
        integrals[:-a-1, a+1:-a-1,a+1:-a-1,a+1, :, :] = \
            rel_coord_a[:, :, 0]*integrals[:-a-1, a+1:-a-1,a+1:-a-1,a, :, :] \
            - rel_coord_point[:, :, 0]*integrals[1:-a, a+1:-a-1,a+1:-a-1,a, :, :] \
            + a/(2*exps_sum.squeeze()) * (
                integrals[:-a-1, a+1:-a-1,a+1:-a-1,a-1, :, :]
                - integrals[1:-a, a+1:-a-1,a+1:-a-1,a-1, :, :]
        )
    
    # Expand the integral array using contracted basis functions (with m = 0):
    # NOTE: Ordering convention for horizontal recursion of integrals
    # axis 0 : a_x (size: m_max)
    # axis 1 : a_y (size: m_max)
    # axis 2 : a_z (size: m_max)
    # axis 3 : b_x (size: angmom_b + 1)
    # axis 4 : b_y (size: angmom_b + 1)
    # axis 5 : b_z (size: angmom_b + 1)

    # Contract basis functions
    temp = integrals.copy()
    integrals = np.zeros((m_max,m_max,m_max, angmom_b+1,angmom_b+1,angmom_b+1))
    integrals[:,:,:, 0,0,0] = np.tensordot(temp, coeffs)[0, :,:,:]

    # Horizontal recursion for one nonzero index i.e. V(120|100)
    for b in range(0, angmom_b):
        # Increment b_x
        integrals[:-1,:,:, b+1,0,0] = integrals[1:,:,:, b,0,0] + rel_dist[:, :, 0]*integrals[:-1,:,:, b,0,0]
        # Increment b_y
        integrals[:,:-1,:, 0,b+1,0] = integrals[:,1:,:, 0,b,0] + rel_dist[:, :, 1]*integrals[:,:-1,:, 0,b,0]
        # Increment b_z
        integrals[:,:,:-1, 0,0,b+1] = integrals[:,:,1:, 0,0,b] + rel_dist[:, :, 2]*integrals[:,:,:-1, 0,0,b]

    # Horizontal recursion for two nonzero indices
    for b in range(0, angmom_b):
        # Increment b_x for all b_y
        integrals[:-1,:,:, b+1,b+1:-b-1,0] =\
            integrals[1:,:,:, b,b+1:-b-1,0] + rel_dist[:, :, 0]*integrals[:-1,:,:, b,b+1:-b-1,0]
        # Increment b_x for all b_z
        integrals[:-1,:,:, b+1,0,b+1:-b-1] =\
            integrals[1:,:,:, b,0,b+1:-b-1] + rel_dist[:, :, 0]*integrals[:-1,:,:, b,0,b+1:-b-1]
        # Increment b_y for all b_x
        integrals[:,:-1,:, b+1:-b-1,b+1,0] =\
            integrals[:,1:,:, b+1:-b-1,b,0] + rel_dist[:, :, 1]*integrals[:,:-1,:, b+1:-b-1,b,0]
        # Increment b_y for all b_z
        integrals[:,:-1,:, 0,b+1,b+1:-b-1] =\
            integrals[:,1:,:, 0,b,b+1:-b-1] + rel_dist[:, :, 1]*integrals[:,:-1,:, 0,b,b+1:-b-1]
        # Increment b_z for all b_x
        integrals[:,:,:-1, b+1:-b-1,0,b+1] =\
            integrals[:,:,1:, b+1:-b-1,0,b] + rel_dist[:, :, 2]*integrals[:,:,:-1, b+1:-b-1,0,b]
        # Increment b_z for all b_y
        integrals[:,:,:-1, 0,b+1:-b-1,b+1] =\
            integrals[:,:,1:, 0,b+1:-b-1,b] + rel_dist[:, :, 2]*integrals[:,:,:-1, 0,b+1:-b-1,b]
        
    # Horizontal recursion for three nonzero indices
    for b in range(0, angmom_b):
        integrals[:-2,:,:, b+1,b+1:-b-1,b+1:-b-1] =\
            integrals[1:-1,:,:, b,b+1:-b-1,b+1:-b-1] + rel_dist[:, :, 0]*integrals[:-2,:,:, b,b+1:-b-1,b+1:-b-1]
        integrals[:-2,:,:, b+1:-b-1,b+1,b+1:-b-1] =\
            integrals[1:-1,:,:, b+1:-b-1,b,b+1:-b-1] + rel_dist[:, :, 1]*integrals[:-2,:,:, b+1:-b-1,b,b+1:-b-1]
        integrals[:-2,:,:, b+1:-b-1,b+1:-b-1,b+1] =\
            integrals[1:-1,:,:, b+1:-b-1,b+1:-b-1,b] + rel_dist[:, :, 2]*integrals[:-2,:,:, b+1:-b-1,b+1:-b-1,b]

    # TODO: Transform to correspond to angular momentum components
    return integrals





