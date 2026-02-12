import numpy as np

def equation(b, s, temp, pH, params):
    k1 = params[0]
    k2 = params[1]
    k3 = params[2]
    k4 = params[3]
    k5 = params[4]
    k6 = params[5]
    k7 = params[6]
    k8 = params[7]
    k9 = params[8]
    k10 = params[9]

    density_term = k1 * b**k2 / (k3 + b**k2)
    substrate_term = k4 * np.exp(-((s - k5) / k6)**2)
    temp_term = np.where(np.abs(temp - k7) < k8, k9, k10)
    ph_term = k10 * np.log(1 + np.abs(pH - k9))

    result = density_term * substrate_term * temp_term * ph_term
    return result
