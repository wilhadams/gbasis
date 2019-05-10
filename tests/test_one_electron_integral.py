"""Test gbasis.one_electron_integral"""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.contractions import ContractedCartesianGaussians
from gbasis.one_electron_integral import OneElectronIntegral, OneElectronCoulomb
import numpy as np
import pytest


def test_one_electron_integral():
    s_type_one = ContractedCartesianGaussians(
        1, np.array([0.5, 1, 1.5]), 0, np.array([1.0]), np.array([0.1])
    )
    s_type_two = ContractedCartesianGaussians(
        1, np.array([1.5, 2, 3]), 0, np.array([3.0]), np.array([0.02])
    )
    OneElectronIntegral.construct_array_contraction(
        s_type_one, s_type_two, np.array([0.0, 0.0, 0.0]), OneElectronCoulomb.boys_func
    )
