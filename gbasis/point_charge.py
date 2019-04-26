"""Module for computing interaction with a point charge (one-electron integrals)."""
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np


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
    def construct_array_contraction(contractions_one, contractions_two, coord_point):
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

        # Should consider swapping a, b for angmom_a < angmom_b

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
            coord_a,
            angmom_a,
            angmoms_a,
            alphas_a,
            coeffs_a,
            norm_a,
            coord_b,
            angmom_b,
            angmoms_b,
            alphas_b,
            coeffs_b,
            norm_b,
        )







