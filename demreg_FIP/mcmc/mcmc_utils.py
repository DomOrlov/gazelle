import numpy as np

def calc_chi2(dn_reg1, dn_in, edn_in):
    chi2 = np.sum(((dn_reg1 - dn_in) / edn_in)**2)
    
    return chi2
