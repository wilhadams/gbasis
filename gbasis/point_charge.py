"""Module for computing interaction with a point charge (one-electron integrals)."""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
from scipy.special import hyp1f1


class PointCharge(BaseTwoIndexSymmetric):
    """Class for calculating the interaction of a point charge with a set of Gaussian contractions.


    Attributes
    ----------
    _axes_contractions : tuple of tuple of ContractedCartesianGaussians
        Sets of contractions associated with each axis of the array.

    Properties
    ----------
    contractions : tuple of ContractedCartesianGaussians
        Contractions that are associated with the first and second indices of the array.

    Methods
    -------
    __init__(self, contractions)
        Initialize.
    construct_array_contraction(self, contraction) :
    """

    @staticmethod
    def boys_func(m, x):
        """Boys function for evaluating the one-electron integral.

        Parameters
        ----------
        m : int
            The helper function index?
        x : np.ndarray(L_b, L_a)
            The weighted interatomic distance, :math:`\mu * R_{AB}^{2}`

        Notes
        -----
        There's some documented instability for hyp1f1, mainly for large values or complex numbers.
        In this case it seems fine, since m should be less than 10 in most cases, and except for
        exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
        problem becomes a precision error in most cases where it was an overflow error before, so
        the values should be close even when they are wrong.
        """

        return hyp1f1(m + 1/2, m + 3/2, -x) / (2 * m + 1)

    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two, coord_point, boys_func=boys_func):
        """Return the evaluations of the point charge interaction for the given contractions.

        Parameters
        ----------
        coord_point : np.ndarray(3,)
            Center of the point charge?
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        """
        # TODO: input checks

        # TODO: Overlap screening

        # TODO: Enforce K_a >= K_b, L_a > L_b
        # Although since a is the inner index for contractions, maybe its K_a <= K_b

        coord_a = contractions_one.coord
        angmom_a = contractions_one.angmom
        angmoms_a = contractions_one.angmom_components
        alphas_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        norm_a = contractions_one.norm
        coord_b = contractions_two.coord
        angmom_b = contractions_two.angmom
        angmoms_b = contractions_two.angmom_components
        alphas_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs
        norm_b = contractions_two.norm

        return _compute_one_elec_integrals(
            coord_point,
            boys_func,
            coord_a,
            angmom_a,
            alphas_a,
            coeffs_a,
            coord_b,
            angmom_b,
            alphas_b,
            coeffs_b,
        )







