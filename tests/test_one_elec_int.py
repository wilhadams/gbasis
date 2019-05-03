from gbasis.contractions import ContractedCartesianGaussians
from gbasis._one_elec_int import _compute_one_elec_integrals
import numpy as np
import pytest
from scipy.special import hyp1f1


def test_compute_one_elec_integrals_s_type():
    # ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps)
    s_type_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    s_type_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.02])
    )
    coord_a = s_type_one.coord
    angmom_a = s_type_one.angmom
    angmoms_a = s_type_one.angmom_components
    exps_a = s_type_one.exps
    coeffs_a = s_type_one.coeffs
    norm_a = s_type_one.norm
    coord_b = s_type_two.coord
    angmom_b = s_type_two.angmom
    angmoms_b = s_type_two.angmom_components
    exps_b = s_type_two.exps
    coeffs_b = s_type_two.coeffs
    norm_b = s_type_two.norm
    s_test = _compute_one_elec_integrals(
        np.array([0., 0., 0.]),      # coord_point
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
    )
    # Check V(m)(000|000) from hand-calculated values.
    v0_000_000 = np.exp(-17/240) * hyp1f1(1/2, 3/2, -701/1200)
    assert np.allclose(
        v0_000_000,
        s_test[0, 0,0,0, 0,0,0, 0, 0]
    )
    v1_000_000 = np.exp(-17 / 240) * hyp1f1(1 + 1 / 2, 1 + 3 / 2, -701 / 1200) / (2 * 1 + 1)
    assert np.allclose(
        v1_000_000,
        s_test[1, 0,0,0, 0,0,0, 0, 0]
    )
    v2_000_000 = np.exp(-17 / 240) * hyp1f1(2 + 1 / 2, 2 + 3 / 2, -701 / 1200) / (2 * 2 + 1)
    assert np.allclose(
        v2_000_000,
        s_test[2, 0,0,0, 0,0,0, 0, 0]
    )

    # Check one-index recursion i.e. V(m)(100|000) using recursion
    v1_100_000 = 1/6 * v1_000_000 - 2/3*v2_000_000
    assert np.allclose(
        v1_100_000,
        s_test[1, 1,0,0, 0,0,0, 0, 0]
    )
    v1_010_000 = 1/6 * v1_000_000 - 14/12*v2_000_000
    assert np.allclose(
        v1_010_000,
        s_test[1, 0,1,0, 0,0,0, 0, 0]
    )
    v1_001_000 = 1/4 * v1_000_000 - 7/4*v2_000_000
    assert np.allclose(
        v1_001_000,
        s_test[1, 0,0,1, 0,0,0, 0, 0]
    )


def test_compute_one_elec_integrals_l1():
    # ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps)
    contr_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.25])
    )
    contr_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.02, 0.01])
    )
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    angmoms_a = contr_one.angmom_components
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    norm_a = contr_one.norm
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    angmoms_b = contr_two.angmom_components
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    norm_b = contr_two.norm
    answer = _compute_one_elec_integrals(
        np.array([0., 0., 0.]),      # coord_point
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
    )
    # Check V(m)(000|000) from hand-calculated values for known s-type values above.
    v0_000_000 = np.exp(-17/240) * hyp1f1(1/2, 3/2, -701/1200)
    assert np.allclose(
        v0_000_000,
        answer[0, 0,0,0, 0,0,0, 0, 0]
    )
    v1_000_000 = np.exp(-17 / 240) * hyp1f1(1 + 1 / 2, 1 + 3 / 2, -701 / 1200) / (2 * 1 + 1)
    assert np.allclose(
        v1_000_000,
        answer[1, 0,0,0, 0,0,0, 0, 0]
    )
    v2_000_000 = np.exp(-17 / 240) * hyp1f1(2 + 1 / 2, 2 + 3 / 2, -701 / 1200) / (2 * 2 + 1)
    assert np.allclose(
        v2_000_000,
        answer[2, 0,0,0, 0,0,0, 0, 0]
    )
    v1_100_000 = 1/6 * v1_000_000 - 2/3*v2_000_000
    assert np.allclose(
        v1_100_000,
        answer[1, 1,0,0, 0,0,0, 0, 0]
    )
    v1_010_000 = 1/6 * v1_000_000 - 14/12*v2_000_000
    assert np.allclose(
        v1_010_000,
        answer[1, 0,1,0, 0,0,0, 0, 0]
    )
    v1_001_000 = 1/4 * v1_000_000 - 7/4*v2_000_000
    assert np.allclose(
        v1_001_000,
        answer[1, 0,0,1, 0,0,0, 0, 0]
    )
    # Check V(m) for other contractions against hand-calculated values
    v0_000_000 = np.exp(-17/440) * hyp1f1(1/2, 3/2, -2041/4400) / (2 * 0 + 1)
    assert np.allclose(
        v0_000_000,
        answer[0, 0,0,0, 0,0,0, 1, 0]
    )
    v1_000_000 = np.exp(-17/440) * hyp1f1(1 + 1/2, 1 + 3/2, -2041/4400) / (2 * 1 + 1)
    assert np.allclose(
        v1_000_000,
        answer[1, 0,0,0, 0,0,0, 1, 0]
    )
    v2_000_000 = np.exp(-17/440) * hyp1f1(2 + 1/2, 2 + 3/2, -2041/4400) / (2 * 2 + 1)
    assert np.allclose(
        v2_000_000,
        answer[2, 0,0,0, 0,0,0, 1, 0]
    )
    v1_100_000 = 1/11 * v1_000_000 - 13/22*v2_000_000
    assert np.allclose(
        v1_100_000,
        answer[1, 1,0,0, 0,0,0, 1, 0]
    )
    v1_010_000 = 1/11 * v1_000_000 - 12/11*v2_000_000
    assert np.allclose(
        v1_010_000,
        answer[1, 0,1,0, 0,0,0, 1, 0]
    )
    v1_001_000 = 3/22 * v1_000_000 - 18/11*v2_000_000
    assert np.allclose(
        v1_001_000,
        answer[1, 0,0,1, 0,0,0, 1, 0]
    )


def test_compute_one_elec_integrals():
    # ContractedCartesianGaussians(angmom, coord, charge, coeffs, exps)
    contr_one = ContractedCartesianGaussians(
        3, np.array([0.5, 1, 1.5]), 0, np.array([1.0, 2.0]), np.array([0.1, 0.25])
    )
    contr_two = ContractedCartesianGaussians(
        2, np.array([1.5, 2, 3]), 0, np.array([3.0, 4.0]), np.array([0.02, 0.01])
    )
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    angmoms_a = contr_one.angmom_components
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    norm_a = contr_one.norm
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    angmoms_b = contr_two.angmom_components
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    norm_b = contr_two.norm
    answer = _compute_one_elec_integrals(
        np.array([0., 0., 0.]),      # coord_point
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
    )
    # Check V(m)(000|000) from hand-calculated values for known s-type values above.
    v0_000_000 = np.exp(-17/240) * hyp1f1(1/2, 3/2, -701/1200)
    assert np.allclose(
        v0_000_000,
        answer[0, 0,0,0, 0,0,0, 0, 0]
    )
    v1_000_000 = np.exp(-17 / 240) * hyp1f1(1 + 1 / 2, 1 + 3 / 2, -701 / 1200) / (2 * 1 + 1)
    assert np.allclose(
        v1_000_000,
        answer[1, 0,0,0, 0,0,0, 0, 0]
    )
    v2_000_000 = np.exp(-17 / 240) * hyp1f1(2 + 1 / 2, 2 + 3 / 2, -701 / 1200) / (2 * 2 + 1)
    assert np.allclose(
        v2_000_000,
        answer[2, 0,0,0, 0,0,0, 0, 0]
    )
    # 1-index vertical recursion
    v0_200_000 = 1/6*answer[0, 1,0,0, 0,0,0, 0,0] - 2/3*answer[1, 1,0,0, 0,0,0, 0,0] \
                 + (answer[0, 0,0,0, 0,0,0, 0,0] - answer[1, 0,0,0, 0,0,0, 0,0])/(2*.12)
    assert np.allclose(
        v0_200_000,
        answer[0, 2,0,0, 0,0,0, 0, 0]
    )
    # 2-index vertical recursion
    v0_110_000 = 1/6*answer[0, 0,1,0, 0,0,0, 0,0] - 2/3*answer[1, 0,1,0, 0,0,0, 0,0]
    assert np.allclose(
        v0_110_000,
        answer[0, 1,1,0, 0,0,0, 0, 0]
    )
    v0_210_000 = 1/6*answer[0, 1,1,0, 0,0,0, 0,0] - 2/3*answer[1, 1,1,0, 0,0,0, 0,0] \
                 + (answer[0, 0,1,0, 0,0,0, 0,0] - answer[1, 0,1,0, 0,0,0, 0,0])/(2*.12)
    assert np.allclose(
        v0_210_000,
        answer[0, 2,1,0, 0,0,0, 0, 0]
    )
    v0_310_000 = 1/6*answer[0, 2,1,0, 0,0,0, 0,0] - 2/3*answer[1, 2,1,0, 0,0,0, 0,0] \
                 + (answer[0, 1,1,0, 0,0,0, 0,0] - answer[1, 1,1,0, 0,0,0, 0,0])*2/(2*.12)
    assert np.allclose(
        v0_310_000,
        answer[0, 3,1,0, 0,0,0, 0, 0]
    )
    v0_220_000 = 1/6*answer[0, 1,2,0, 0,0,0, 0,0] - 2/3*answer[1, 1,2,0, 0,0,0, 0,0] \
                 + (answer[0, 0,2,0, 0,0,0, 0,0] - answer[1, 0,2,0, 0,0,0, 0,0])/(2*.12)
    assert np.allclose(
        v0_220_000,
        answer[0, 2,2,0, 0,0,0, 0, 0]
    )
    v0_022_000 = 0.25*answer[0, 0,2,1, 0,0,0, 0,0] - 1.75*answer[1, 0,2,1, 0,0,0, 0,0] \
                 + (answer[0, 0,2,0, 0,0,0, 0,0] - answer[1, 0,2,0, 0,0,0, 0,0])/(2*.12)
    assert np.allclose(
        v0_022_000,
        answer[0, 0,2,2, 0,0,0, 0, 0]
    )
    # Check for no out-of-bounds assignments
    m_max = angmom_a + angmom_b + 1
    for i in range(1, m_max-1):
        assert np.allclose(
            answer[0, i,m_max-i,0, 0,0,0, 0, 0],
            0.0
        )
        assert np.allclose(
            answer[0, m_max-i,i,0, 0,0,0, 0, 0],
            0.0
        )
        assert np.allclose(
            answer[0, 0,i,m_max-i, 0,0,0, 0, 0],
            0,0
        )
        assert np.allclose(
            answer[0, 0,m_max-i,i, 0,0,0, 0, 0],
            0.0
        )
        assert np.allclose(
            answer[0, i,0,m_max-i, 0,0,0, 0, 0],
            0.0
        )
        assert np.allclose(
            answer[0, m_max-i,0,i, 0,0,0, 0, 0],
            0.0
        )
    '''
    # 3-index vertical recursion
    v0_111_000 = 1/6*answer[0, 0,1,1, 0,0,0, 0,0] - 2/3*answer[0, 1,1,1, 0,0,0, 0,0]
    assert np.allclose(
        v0_111_000,
        answer[0, 1,1,1, 0,0,0, 0, 0]
    )
    '''
    # Check values for different contractions
    v0_200_000 = 1/11*answer[0, 1,0,0, 0,0,0, 1,0] - 13/22*answer[1, 1,0,0, 0,0,0, 1,0] \
                 + (answer[0, 0,0,0, 0,0,0, 1,0] - answer[1, 0,0,0, 0,0,0, 1,0])/(2*.11)
    assert np.allclose(
        v0_200_000,
        answer[0, 2,0,0, 0,0,0, 1, 0]
    )
    v0_020_000 = 1/11*answer[0, 0,1,0, 0,0,0, 1,0] - 12/11*answer[1, 0,1,0, 0,0,0, 1,0] \
                 + (answer[0, 0,0,0, 0,0,0, 1,0] - answer[1, 0,0,0, 0,0,0, 1,0])/(2*.11)
    assert np.allclose(
        v0_020_000,
        answer[0, 0,2,0, 0,0,0, 1, 0]
    )
    v0_002_000 = 3/22*answer[0, 0,0,1, 0,0,0, 1,0] - 18/11*answer[1, 0,0,1, 0,0,0, 1,0] \
                 + (answer[0, 0,0,0, 0,0,0, 1,0] - answer[1, 0,0,0, 0,0,0, 1,0])/(2*.11)
    assert np.allclose(
        v0_002_000,
        answer[0, 0,0,2, 0,0,0, 1, 0]
    )


def test_compute_one_elec_integrals_speed():
    import time
    contr_one = ContractedCartesianGaussians(
        5, np.array([0.5, 1, 1.5]), 0, np.random.random_sample((10,)), np.random.random_sample((10,))
    )
    contr_two = ContractedCartesianGaussians(
        5, np.array([1.5, 2, 3]), 0, np.random.random_sample((10,)), np.random.random_sample((10,))
    )
    t0 = time.time()
    coord_a = contr_one.coord
    angmom_a = contr_one.angmom
    angmoms_a = contr_one.angmom_components
    exps_a = contr_one.exps
    coeffs_a = contr_one.coeffs
    norm_a = contr_one.norm
    coord_b = contr_two.coord
    angmom_b = contr_two.angmom
    angmoms_b = contr_two.angmom_components
    exps_b = contr_two.exps
    coeffs_b = contr_two.coeffs
    norm_b = contr_two.norm
    answer = _compute_one_elec_integrals(
        np.array([0., 0., 0.]),      # coord_point
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
    )
    t1 = time.time()
    print("{} sec".format(t1-t0))

