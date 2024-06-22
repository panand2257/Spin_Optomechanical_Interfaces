import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import *
T = tensor

kb = 1.380649e-23
h = 6.62607015e-34

#Heralding Fidelity from Stefan's code

def λ_1(T_a, gamma, n_th):
    return 0.25*(1/T_a + gamma*(n_th+0.5))
def P(T,g,T_a,gamma,n_th):
    return T/T_a * (g/2/λ_1(T_a,gamma,n_th))**2 * (1+gamma*T/4*(3*n_th+1))
def F(T,g,T_a,gamma,n_th):
    return (1-np.exp(-C(g,T_a,gamma,n_th)*T))**2 / (C(g,T_a,gamma,n_th)*T)**2 / (1+gamma*T/4*(3*n_th+1))
def C(g,T_a,gamma,n_th):
    return 8*g**2 / (1/T_a+gamma*(n_th+1/2))  +  gamma/2*(3*n_th+1)

#Constraints on T_pump
# T_a << T_pump << 1/gamma_b
# T_a << 1/( np.sqrt(n_pump) * g_om )


def fid_herald(n_pump, T_pump, temp_bath, f_mech, Q_mech, f_opt, Q_opt, g_om):
    gamma_a = f_opt/Q_opt #in Hz
    gamma_m = 2*f_mech/Q_mech #in Hz
    n_th = kb * temp_bath /(h * f_mech)
    gamma_b = gamma_m * (n_th + 1/2) #in Hz
    Ta = 1/gamma_a #in sec
    #T_min = 1/(gamma_a + gamma_b)
    g_om_ph = np.sqrt(n_pump) * g_om #input g_om in Hz in COMSOL

    f_herald_val = F(T_pump, g_om_ph, Ta, gamma_m, n_th)

    # if ( (0.99*Ta < T_pump) and ( T_pump < 1/gamma_b) and (Ta < 1/g_om_ph)):
    #     f_herald_val = F(T_pump, g_om_ph, Ta, gamma_m, n_th)
    
    # else: 
    #     f_herald_val = np.nan

    return f_herald_val


def fid_swap (temp_bath, Q_mech, f_mech, g_sm): 
    #Problem Parameters
    k_per_h = 20836617636.1328 # Hz per K
    n_per_T = k_per_h/f_mech # photons per K
    N_opt = 3
    N_mech = 3

    b = destroy(N_mech)
    s = destroy(2)

    Id_opt = identity(N_opt)
    Id_mech = identity(N_mech)
    Id_spin = identity(2)

    #Initial states for Swap protocol
    psi0_opt = basis(N_opt, 0)
    psi0_mech = basis(N_mech, 1)
    psi0_spin = basis(2, 0)
    psi0 = T(psi0_opt, psi0_mech, psi0_spin)

    T1e = 1
    Temp = temp_bath

    nT = n_per_T*Temp                              
    gamma = 4 * np.pi * f_mech/Q_mech

    time_scale_swap = 10000
    t_up_swap = 0.238/g_sm  # for pi rotation
    tlist_swap = np.linspace(0,t_up_swap,time_scale_swap)

    #detuning parameters
    delta_b = 0
    delta_s = 0

    #Create Hamiltonian in interaction picture
    H_b = 2 * np.pi * delta_b * T(Id_opt , b.dag() * b, Id_spin)
    H_s = 2 * np.pi * delta_s * T(Id_opt , Id_mech, s.dag() * s)
    H_sm = 2 * np.pi * g_sm * (T(Id_opt,b, s.dag()) + T(Id_opt,b.dag(), s)) 

    H_swap = H_b + H_s + H_sm

    #Create collapse operators
    cb = (gamma/2*(nT+1))**0.5 * T(Id_opt,b, Id_spin)       #phonon decay
    cb1 = (gamma/2*nT)**0.5 * T(Id_opt,b.dag(), Id_spin)    #phonon absorption
    cs = (2*np.pi/T1e)**0.5 * T(Id_opt, Id_mech, s)         #spin decay

    c_op_list_swap = [cb, cb1, cs]
    me_swap = mesolve(H_swap, psi0, tlist_swap, c_op_list_swap, [])

    swap_state = me_swap.states[time_scale_swap - 1]
    swap_spin_state_dm = swap_state.ptrace(2)

    fid_swap_val = (swap_spin_state_dm * (qt.ket2dm(basis(2, 1)))).tr()
    
    return (fid_swap_val)


def fid_total(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    f_init = 0.99
    f_her = fid_herald(n_pump, T_pump, temp_bath, f_mech, Q_mech, f_opt, Q_opt, g_om)
    f_swap = fid_swap(temp_bath, Q_mech, f_mech, g_sm)
    f_total = f_init * f_her * f_swap

    return (f_total)

def fid_entanglement(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    f_init = 0.99
    f_her = fid_herald(n_pump, T_pump, temp_bath, f_mech, Q_mech, f_opt, Q_opt, g_om)
    f_swap = fid_swap(temp_bath, Q_mech, f_mech, g_sm)
    f_total = (f_init**2) * f_her * (f_swap**2)

    f_ent = f_total

    return (f_ent)

def heralding_rate(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    gamma_a = f_opt/Q_opt #in Hz
    gamma_m = 2*f_mech/Q_mech #in Hz
    n_th = kb * temp_bath /(h * f_mech)
    gamma_b = gamma_m * (n_th + 1/2) #in Hz
    Ta = 1/gamma_a #in sec
    g_om_ph = np.sqrt(n_pump) * g_om #input g_om in Hz from COMSOL

    T_reset = 2.0e-7 #sec
    T_swap = 0.238/g_sm #sec

    T_protocol =  T_reset + T_pump + T_swap

    hr = P(T_pump, g_om_ph, Ta, gamma_m, n_th)/T_protocol

    return hr

def ent_rate(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    gamma_a = f_opt/Q_opt #in Hz
    gamma_m = 2*f_mech/Q_mech #in Hz
    n_th = kb * temp_bath /(h * f_mech)
    gamma_b = gamma_m * (n_th + 1/2) #in Hz
    Ta = 1/gamma_a #in sec
    g_om_ph = np.sqrt(n_pump) * g_om #input g_om in Hz from COMSOL

    T_reset = 2.0e-7 #sec
    T_swap = 0.238/g_sm #sec

    T_protocol =  T_reset + T_pump + T_swap

    er = 2*P(T_pump, g_om_ph, Ta, gamma_m, n_th)*(1-P(T_pump, g_om_ph, Ta, gamma_m, n_th))/T_protocol

    return er

def fom(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    F = fid_entanglement(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)
    R = ent_rate(n_pump, T_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)

    T2 = 0.9e-3

    C = 1 - F * np.exp(- 1 / (T2 * R))

    return (F, R, C)

def fom_sweep(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):

    T_pump = np.geomspace(1e-9, 1e-5, 70)
    C_sweep = []
    F_sweep = []
    R_sweep = []
    for tpump in T_pump:
        C_sweep.append(fom(n_pump, tpump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[2])
        F_sweep.append(fom(n_pump, tpump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[0])
        R_sweep.append(fom(n_pump, tpump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[1])

    return (F_sweep, R_sweep, C_sweep)

def opt_par(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):


    arg_opt = np.argmin(np.array(fom_sweep(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[2]))
    fid_opt = fom_sweep(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[0][arg_opt]
    rate_opt = fom_sweep(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[1][arg_opt]
    cost_opt = fom_sweep(n_pump, temp_bath, Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm)[2][arg_opt]

    pump_opt = np.geomspace(1e-9, 1e-5, 70)[arg_opt]

    return (fid_opt, rate_opt, cost_opt, pump_opt)
