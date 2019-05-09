"""Module for computing one-electron integrals."""
import abc
from gbasis._one_elec_int import _compute_one_elec_integrals
from gbasis.base_two_symm import BaseTwoIndexSymmetric
from gbasis.contractions import ContractedCartesianGaussians
import numpy as np
from scipy.special import hyp1f1


class OneElectronIntegral(BaseTwoIndexSymmetric):
    """General class for calculating one electron integrals (interaction with a point charge).

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
    boys_func
        Boys function for evaluating the one-electron integral.
    construct_array_contraction(self, contraction) :

    """

    @staticmethod
    @abc.abstractmethod
    def boys_func(m, x):
        """Boys function for evaluating the one-electron integral.

        Parameters
        ----------
        m : int
            Differentiation order of the helper function.
        x : np.ndarray(L_b, L_a)
            The weighted interatomic distance, :math:`\\mu * R_{AB}^{2}`.

        Notes
        -----
        There's some documented instability for hyp1f1, mainly for large values or complex numbers.
        In this case it seems fine, since m should be less than 10 in most cases, and except for
        exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
        problem becomes a precision error in most cases where it was an overflow error before, so
        the values should be close even when they are wrong.

        """

    @staticmethod
    def construct_array_contraction(contractions_one, contractions_two, coord_point, boys_func):
        """Return the evaluations of the point charge interaction for the given contractions.

        Parameters
        ----------
        contractions_one : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the first index of
            the array.
        contractions_two : ContractedCartesianGaussians
            Contracted Cartesian Gaussians (of the same shell) associated with the second index of
            the array.
        coord_point : np.ndarray(3,)
            Center of the point charge.
        boys_func :
            Boys function used to evaluate the one-electron integral.

        """

        # TODO: input checks for coord_point, boys_func

        # TODO: Overlap screening?

        # TODO: Enforce K_a >= K_b, L_a > L_b
        # Although since a is the inner index for contractions, maybe its K_a <= K_b

        coord_a = contractions_one.coord
        angmom_a = contractions_one.angmom
        angmoms_a = contractions_one.angmom_components
        exps_a = contractions_one.exps
        coeffs_a = contractions_one.coeffs
        norm_a = contractions_one.norm
        coord_b = contractions_two.coord
        angmom_b = contractions_two.angmom
        angmoms_b = contractions_two.angmom_components
        exps_b = contractions_two.exps
        coeffs_b = contractions_two.coeffs
        norm_b = contractions_two.norm

        integrals = _compute_one_elec_integrals(
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
        )

        # Generate output array
        # Ordering for output array:
        # axis 0 : index for segmented contractions of contraction one
        # axis 1 : angular momentum vector of contraction one (in the same order as angmoms_a)
        # axis 2 : index for segmented contractions of contraction two
        # axis 3 : angular momentum vector of contraction two (in the same order as angmoms_b)
        output = np.zeros(
            (len(coeffs_a[0]), contractions_one.num_contr, len(coeffs_b[0]), contractions_two.num_contr)
        )

        for i,comp_a in enumerate(angmoms_a):
            a_x, a_y, a_z = comp_a
            for j,comp_b in enumerate(angmoms_b):
                b_x, b_y, b_z = comp_b
                output[:, i, :, j] = integrals[a_x, a_y, a_z, b_x, b_y, b_z, :, :]

        return output


def one_electron_integral_basis_cartesian(basis, coord_point, boys_func):
    """Return the one-electron integrals of the basis set in the Cartesian form.

    """
    return OneElectronIntegral(basis).construct_array_cartesian(coord_point=coord_point, boys_func=boys_func)


def one_electron_integral_basis_spherical(basis, coord_point, boys_func):
    """Return the one-electron integrals of the basis set in the spherical form.

    """
    return OneElectronIntegral(basis).construct_array_spherical(coord_point=coord_point, boys_func=boys_func)


def one_electron_integral_spherical_lincomb(basis, transform, coord_point, boys_func):
    """Return the one-electron integrals of the linear combination of the basis set in the spherical form.

    """
    return OneElectronIntegral(basis).construct_array_spherical_lincomb(transform, coord_point=coord_point, boys_func=boys_func)


class OneElectronCoulomb(OneElectronIntegral):
    """Evaluate one-electron Coulomb interaction integrals.

    """
    @staticmethod
    def boys_func(m, x):
        """Boys function for evaluating the one-electron integral.

        Parameters
        ----------
        m : int
            Differentiation order of the helper function.
        x : np.ndarray(L_b, L_a)
            The weighted interatomic distance, :math:`\\mu * R_{AB}^{2}`.

        Notes
        -----
        There's some documented instability for hyp1f1, mainly for large values or complex numbers.
        In this case it seems fine, since m should be less than 10 in most cases, and except for
        exceptional cases the input, while negative, shouldn't be very large. In scipy > 0.16, this
        problem becomes a precision error in most cases where it was an overflow error before, so
        the values should be close even when they are wrong.

        """

        return hyp1f1(m + 1/2, m + 3/2, -x) / (2 * m + 1)
