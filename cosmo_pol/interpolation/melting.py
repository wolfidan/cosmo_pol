# -*- coding: utf-8 -*-

"""melting.py: Computes mass concentrations of melted snow and melted graupel
as well as the wet fractions for both melted hydrometeor types"""

__author__ = "Daniel Wolfensberger"
__copyright__ = "Copyright 2017, COSMO_POL"
__credits__ = ["Daniel Wolfensberger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Daniel Wolfensberger"
__email__ = "daniel.wolfensberger@epfl.ch"


# Global imports
import numpy as np
np.seterr(divide='ignore') # Disable divide by zero error

def melting(subradial):

    """
    Checks an interpolated subradial for melting and if any melting should
    occur, apply the melting
    Args:
        subradial: Radial class instance containing all data about the
            subradial which should be checked for melting
    Returns:
        subradial: returns the original subradial but with melting applied
            (if necessary)
    """

    # This vector allows to know if a given beam has some melting particle
    has_melting = False

    T =  subradial.values['T']
    QR = subradial.values['QR_v']
    QS = subradial.values['QS_v']
    QG = subradial.values['QG_v']

    QS_QG = QS + QG

    mask_ml = np.logical_and(QR > 0, QS_QG > 0)

    if not np.any(mask_ml):
        has_melting = False
        subradial.values['QmS_v'] = np.zeros((len(T)))
        subradial.values['QmG_v'] = np.zeros((len(T)))
        subradial.values['fwet_mS'] = np.zeros((len(T)))
        subradial.values['fwet_mG'] = np.zeros((len(T)))
    else:
        has_melting = True
        # List of bins where melting takes place

        # Retrieve rain and dry solid concentrations
        QR_in_ml = subradial.values['QR_v'][mask_ml]
        QS_in_ml = subradial.values['QS_v'][mask_ml]
        QG_in_ml = subradial.values['QG_v'][mask_ml]

        QS_QG_in_ml = QS_in_ml + QG_in_ml


        subradial.values['QmS_v'] = np.zeros((len(T)))
        subradial.values['QmS_v'][mask_ml] = (QS_in_ml + QR_in_ml *
                                                (QS_in_ml / QS_QG_in_ml))
        subradial.values['QmG_v'] = np.zeros((len(T)))
        subradial.values['QmG_v'][mask_ml] = (QG_in_ml + QR_in_ml *
                                                (QG_in_ml / QS_QG_in_ml))

        mask_with_melting = np.logical_or(subradial.values['QmS_v'] > 0,
                                          subradial.values['QmG_v'] > 0)

        subradial.values['QS_v'][mask_with_melting] = 0
        subradial.values['QG_v'][mask_with_melting] = 0
        subradial.values['QR_v'][mask_with_melting] = 0

        # Wet fraction = mass coming from rain / total mass
        subradial.values['fwet_mS'] = np.zeros((len(T)))
        subradial.values['fwet_mS'][mask_ml] = ((QR_in_ml * QS_in_ml / QS_QG_in_ml)
                                            / subradial.values['QmS_v'][mask_ml])

        subradial.values['fwet_mG'] = np.zeros((len(T)))
        subradial.values['fwet_mG'][mask_ml] = ((QR_in_ml * QG_in_ml / QS_QG_in_ml)
                                            / subradial.values['QmG_v'][mask_ml])


    # Add an attribute to the beam that specifies if any melting occurs
    subradial.has_melting = has_melting
    subradial.mask_ml = mask_ml

    return subradial
