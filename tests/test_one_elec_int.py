from gbasis.contractions import ContractedCartesianGaussians
from gbasis._one_elec_int import _compute_one_elec_integrals
import numpy as np
import pytest
from scipy.special import hyp1f1


def test_compute_one_elec_integrals():
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
    assert np.allclose(
        np.exp(np.array([-1/60, -1/60, -0.0375])) *
        hyp1f1(1/2, 3/2, np.array([-(1/20+1/300), -(16/100 + 1/300), -0.3675])),
        s_test[0, 0,0,0, 0,0,0, :, 0, 0]
    )
    assert np.allclose(
        np.exp(np.array([-1 / 60, -1 / 60, -0.0375])) *
        (hyp1f1(1 / 2 + 1, 3 / 2 + 1, np.array([-(1 / 20 + 1 / 300), -(16 / 100 + 1 / 300), -0.3675])) / (2 * 1 + 1)),
        s_test[1, 0,0,0, 0,0,0, :, 0, 0]
    )
    assert np.allclose(
        np.exp(np.array([-1 / 60, -1 / 60, -0.0375])) *
        (hyp1f1(1 / 2 + 2, 3 / 2 + 2, np.array([-(1 / 20 + 1 / 300), -(16 / 100 + 1 / 300), -0.3675])) / (2 * 2 + 1)),
        s_test[2, 0,0,0, 0,0,0, :, 0, 0]
    )


