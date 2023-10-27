import numpy as np
import matplotlib.pyplot as plt
from fidelity_analytical import *


# Using readlines()
file1 = open('sample.txt', 'r')
Lines = file1.readlines()


g_om_iter = []
g_sm_iter = []
f_opt_iter = []
f_mech_iter = []
Q_opt_iter = []
Q_mech_iter = []

for L in Lines[1:]:

    data = L.split()
    g_om_iter.append(float(data[3]))
    g_sm_iter.append(float(data[4]))
    f_opt_iter.append(float(data[5]))
    f_mech_iter.append(float(data[6]))
    Q_opt_iter.append(float(data[7]))
    Q_mech_iter.append(float(data[8]))

pump = [10**0, 10**0.5, 10**1, 10**1.5, 10**2, 10**2.5, 10**3, 10**3.5, 10**4]
F_iter_pump = []
R_iter_pump = []
#len(g_om_iter)


for iter in np.arange(1):

    F_pump = []
    R_pump = []

    for n_pump in pump:

        f = fid_total(n_pump, 4, Q_opt_iter[iter], Q_mech_iter[iter], f_opt_iter[iter], f_mech_iter[iter], g_om_iter[iter], g_sm_iter[iter])
        r = heralding_rate(n_pump, 4, Q_opt_iter[iter], Q_mech_iter[iter], f_opt_iter[iter], f_mech_iter[iter], g_om_iter[iter], g_sm_iter[iter])
        F_pump.append(f)
        R_pump.append(r)

    F_iter_pump.append(F_pump)
    R_iter_pump.append(R_pump)


plt.plot(np.log10(np.array(pump)), np.array(F_iter_pump[0]), 'o', color = 'red')
plt.ylabel("Fidelity")
plt.xlabel("log(n_pump)")
plt.show()














