#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from cosmo_pol.constants import constants

'''
Computes mass concentrations of melted snow and melted graupel as well as the
wet fractions for both melted hydrometeor types
'''
def melting(list_beams):
    # This vector allows to know if a given beam has some melting particle
    # i.e. temperatures fall between MIN_T_MELT and MAX_T_MELT
    has_melting = np.zeros((len(list_beams)))

    for i,beam in enumerate(list_beams): # Loop on subbeams
        T = beam.values['T']
        if not np.any(np.logical_and(T < constants.MAX_T_MELT,
                                         T > constants.MIN_T_MELT)):
            has_melting[i] = False
        else:
            has_melting[i] = True
            # List of bins where melting takes place
            mask = np.logical_and(T < constants.MAX_T_MELT,
                                 T > constants.MIN_T_MELT)
            # Retrieve rain and dry solid concentrations
            QR_in_ml = beam.values['QR_v'][mask]
            QS_in_ml = beam.values['QS_v'][mask]
            QG_in_ml = beam.values['QG_v'][mask]

            # See Jung et al. 2008
            frac_mS = np.minimum(QS_in_ml / QR_in_ml, QR_in_ml / QS_in_ml)**0.3

            frac_mG = np.minimum(QG_in_ml / QR_in_ml, QR_in_ml / QG_in_ml)**0.3

            # Correct QR: 3 contributions: from rainwater, from snow and from graupel
            beam.values['QR_v'][mask] = ((1 - frac_mS - frac_mG) * QR_in_ml
                                        + (1 - frac_mS) * QS_in_ml
                                        + (1 - frac_mG) * QG_in_ml)


            # Add QmS and QmG
            beam.values['QmS_v'] = np.zeros((len(T)))
            beam.values['QmS_v'][mask] = frac_mS * (QR_in_ml + QS_in_ml)
            beam.values['QmG_v'] = np.zeros((len(T)))
            beam.values['QmG_v'][mask] = frac_mG * (QR_in_ml + QG_in_ml)

            # Add wet fractions
            beam.values['fwet_mS'] = np.zeros((len(T)))
            beam.values['fwet_mS'][mask] = QS_in_ml / (QR_in_ml + QS_in_ml)

            beam.values['fwet_mG'] = np.zeros((len(T)))
            beam.values['fwet_mG'][mask] = QG_in_ml / (QR_in_ml + QG_in_ml)

            # Remove dry hydrometeors where melting is taking place
            beam.values['QS_v'][mask] = 0
            beam.values['QG_v'][mask] = 0

    return list_beams


if __name__ == '__main__':
    import pickle
    beams = pickle.load(open('/data/cosmo_pol/ex_beams_rhi.txt','rb'))
    plt.plot(beams[7].values['QS_v'])
    l = melting(beams)
    plt.plot(l[7].values['QS_v'])
    plt.plot(l[7].values['QmS_v'])