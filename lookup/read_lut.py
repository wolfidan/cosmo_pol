import pickle
import numpy as np
import os
import gzip

FOLDER_LUT=os.path.dirname(os.path.realpath(__file__))

from cosmo_pol.lookup.compute_lut_sz import FREQUENCIES, HYDROM_TYPES

def get_lookup_tables(scheme, freq, scattering_method):
    lut_sz = []
    if scattering_method == 'tmatrix_old':
        folder_lut = FOLDER_LUT +'/final_lut/'
    elif scattering_method == 'tmatrix_new':
        folder_lut = FOLDER_LUT +'/final_lut_quad/'
    elif scattering_method == 'dda':
        folder_lut = FOLDER_LUT +'/final_lut_dda/'

    try:
        lut_sz = pickle.load(gzip.open(folder_lut+'all_luts_SZ_f_'+str(freq).replace('.','_')+'_'+str(scheme)+'mom.pz','r'))
    except:
        msg = """
        No set of polarimetric lookup tables corresponding to this specific
        frequency and scattering method was found!
        """
        raise IOError(msg)
        # TODO
#        print('Creating a set of lookup tables by interpolating from the two closest frequencies '+
#        'This might take a while (but will be done only once)')
#        interpolate_lookup_tables(scheme,freq)
#        return get_lookup_tables(scheme,freq)
    return lut_sz

def interpolate_lookup_tables(scheme,freq):
    # TODO
    idx_closest=np.searchsorted(FREQUENCIES, freq)
    # Radar variables
    lut_1=pickle.load(gzip.open(FOLDER_LUT+ "all_luts_rad_f_"+str(FREQUENCIES[idx_closest-1]).replace('.','_')+'_'+scheme+".pz",'r'))
    lut_2=pickle.load(gzip.open(FOLDER_LUT+ "all_luts_rad_f_"+str(FREQUENCIES[idx_closest]).replace('.','_')+'_'+scheme+".pz",'r'))
    lut_rad=dict.fromkeys(HYDROM_TYPES)
    for h in lut_1.keys():
        lut_rad[h]={}
        for v in lut_1[h].keys():
            new_lut = lut_1

            tab1=np.asarray(lut_1[h][v].value_table)
            tab2=np.asarray(lut_2[h][v].value_table)
            interp_values=tab1+(tab2-tab1)/(FREQUENCIES[idx_closest]-FREQUENCIES[idx_closest-1])*(freq-FREQUENCIES[idx_closest-1])
            new_lut.setValueTable(interp_values.tolist())
            freq
            lut_rad[h][v]=new_lut
    # RCS
    lut_1=pickle.load(gzip.open(FOLDER_LUT+ "all_luts_rcs_f_"+str(FREQUENCIES[idx_closest-1]).replace('.','_')+'_'+scheme+".pz",'r'))
    lut_2=pickle.load(gzip.open(FOLDER_LUT+ "all_luts_rcs_f_"+str(FREQUENCIES[idx_closest]).replace('.','_')+'_'+scheme+".pz",'r'))
    lut_rcs=dict.fromkeys(HYDROM_TYPES)

    for h in lut_1.keys():
        new_lut = lut_1

        tab1=np.asarray(lut_1[h].value_table)
        tab2=np.asarray(lut_2[h].value_table)
        interp_values=tab1+(tab2-tab1)/(FREQUENCIES[idx_closest]-FREQUENCIES[idx_closest-1])*(freq-FREQUENCIES[idx_closest-1])
        new_lut.setValueTable(interp_values.tolist())

        lut_rcs[h]=new_lut

    pickle.dump(lut_rcs,gzip.open(FOLDER_LUT+'all_luts_rcs_f_'+str(freq).replace('.','_')+'_'+scheme+'.pz','wb'))
    pickle.dump(lut_rad,gzip.open(FOLDER_LUT+'all_luts_rad_f_'+str(freq).replace('.','_')+'_'+scheme+'.pz','wb'))


def float_to_uint16(x, use_log=True):
    eps = np.finfo(float).eps
    log_offset = 0
    if use_log:
        log_offset=eps+np.nanmin(x)
        x=np.log10(x+log_offset)

    x[x==np.Inf] = float('nan')

    int_min = -32768
    int_max = 32767
    float_min = np.nanmin(x)
    float_max = np.nanmax(x)

    x_d=np.zeros(x.shape)
    x_d[~np.isnan(x)]=(int_min+1)+(x[~np.isnan(x)]-float_min)/(float_max-float_min)*(int_max-(int_min+1))
    x_d[np.isnan(x)]=int_min
    x_d = x_d.astype('int16')

    out={'data':x_d,'use_log':use_log,'log_offset':log_offset,'range':[float_min,float_max]}

    return out

def uint16_to_float(in_dic):
    int_min = -32768
    int_max = 32767
    float_min = in_dic['range'][0]
    float_max = in_dic['range'][1]

    x_d=in_dic['data'].astype('float64')
    x=np.zeros(x_d.shape)
    x[x_d!=int_min]=(float_min)+(x_d[x_d!=int_min]-(int_min+1))/(int_max-(int_min+1))*(float_max-float_min)
    x[x_d==int_min]=float('nan')

    if in_dic['use_log']:
        x=10**x-in_dic['log_offset']
    return x


if __name__ == '__main__':
    a=get_lookup_tables('1mom',5.6)
#
#