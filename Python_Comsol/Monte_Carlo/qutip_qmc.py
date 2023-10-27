import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import *
T = tensor

if __name__ == '__main__':  
    #Problem Parameters
    Fopt = 193e12 #Hz
    Q_opt = 5.0e4
    T_opt = Q_opt/Fopt
    Q_mech = 1.0e5 
    Fmech = 4.7578e9/2/np.pi # 2.7 GHz omega
    k_per_h = 20836617636.1328 # Hz per K
    n_per_T_mech = k_per_h/Fmech # phonons per K
    n_per_T_opt = k_per_h/Fopt
    Delta_p = 0
    Delta_sig = 0

    k_per_h = 20.84e9 # MHz per K
    n_per_T = k_per_h/Fmech # photons per K


    N_opt = 3
    N_mech = 3
    a = destroy(N_opt)
    b = destroy(N_mech)
    s = destroy(2)

    Id_opt = identity(N_opt)
    Id_mech = identity(N_mech)
    Id_spin = identity(2)

    #Initial states for cooling
    psi0_opt_cool = qt.ket2dm(basis(N_opt, 0))
    #psi0_mech_cool = thermal_dm(N_mech, 16.666)
    #psi0_spin_cool = thermal_dm(2, 186.94)

    # psi0_opt_cool = qt.ket2dm(basis(N_opt, 1))
    psi0_mech_cool = qt.ket2dm(basis(N_mech, 2))
    psi0_spin_cool = qt.ket2dm(basis(2, 1))

    psi0_cool = T(psi0_opt_cool, psi0_mech_cool, psi0_spin_cool)


    #Initial states for Herlading + Swap protocol
    psi0_opt = basis(N_opt, 0)
    psi0_mech = basis(N_mech, 0)
    psi0_spin = basis(2, 0)


    psi0 = T(psi0_opt, psi0_mech, psi0_spin)
    g_om = 2.0e7
    g_om_cooling = 3.0e8
    g_sm = 1.0e7
    a_bar = 1.0
    Temp = 4 #Kelvin
    T1e = 1000e-6

    T_pump = 1.0e-8 #sec

    time_scale_cool = 10000
    t_up_cool = 5.0e-7
    del_t_cool = t_up_cool/time_scale_cool

    tlist_cool = np.linspace(0,t_up_cool,time_scale_cool)
    numb_cool = 1000


    

    time_scale = 8000
    t_up = 1.0e-8
    del_t = t_up/time_scale

    tlist = np.linspace(0,t_up,time_scale)
    numb = 1000


    #detuning parameters
    delta_a = 0
    delta_b = 0
    delta_s = 0

    #Create Hamiltonian in interaction picture
    H_a = 2 * np.pi * delta_a * T(a.dag() * a , Id_mech, Id_spin)
    H_b = 2 * np.pi * delta_b * T(Id_opt , b.dag() * b, Id_spin)
    H_s = 2 * np.pi * delta_s * T(Id_opt , Id_mech, s.dag() * s)

    H_om = 2 * np.pi * g_om * a_bar * (T(a.dag(),b.dag(), Id_spin) + T(a, b, Id_spin)) # after rotating wave approximation

    H_om_red = 2 * np.pi * g_om_cooling * a_bar * (T(a.dag(),b, Id_spin) + T(a, b.dag(), Id_spin)) # red-detuning for phonon-spin cooling

    H_sm = 2 * np.pi * g_sm * (T(Id_opt,b, s.dag()) + T(Id_opt,b.dag(), s)) 

    H = H_a + H_b + H_s + H_om + H_sm
    H_cool = H_a + H_b + H_s + H_om_red + H_sm 

    #Create collapse operators
    nT = n_per_T*Temp                               # thermal photon count
    gamma = 4*np.pi*Fmech/Q_mech

    lambda_l = 1/4*(1/T_opt + gamma*(nT + 0.5))
    lambda_h01 = gamma * (0.75*nT + 0.25)
    g_til = (2**0.5) * g_om 
    C = g_til**2/lambda_l + 2*lambda_h01

    P_a_then_bad_th = (T_pump + (C**-1)*(np.exp(-C * T_pump) - 1))/T_opt * (0.5*g_om/lambda_l)*(0.5*g_om/lambda_l)

    P_first_a_th = T_pump/T_opt * (0.5*g_om/lambda_l)*(0.5*g_om/lambda_l)
    #P_a_then_bad = 
    P_ba_th = 0.5 * T_pump**2/T_opt * gamma*(nT+1)/2 * (0.5*g_om/lambda_l)*(0.5*g_om/lambda_l)
    P_b1a_th = 0.5 * T_pump**2/T_opt * gamma*(nT)/2 * 2*(0.5*g_om/lambda_l)*(0.5*g_om/lambda_l)
    p_lim_b = gamma*(nT+1)/2
    #* (0.5*g_om/lambda_l)*(0.5*g_om/lambda_l)  
    p_lim_b1 = gamma*nT/2                   

    ca = (2*np.pi/T_opt)**0.5 * T(a, Id_mech, Id_spin)  # cavity photon decay
    cb = (gamma/2*(nT+1))**0.5 * T(Id_opt,b, Id_spin)      # phonon decay
    cb1 = (gamma/2*nT)**0.5 * T(Id_opt,b.dag(), Id_spin)  #phonon absorption
    cs = (2*np.pi/T1e)**0.5 * T(Id_opt, Id_mech, s) #spin decay

    c_op_list = [ca, cb, cb1, cs]

    spin_rho_11 = (T(Id_opt, Id_mech, s).dag() * T(Id_opt, Id_mech, s))
    spin_rho_10 = (T(Id_opt, Id_mech, s).dag())
    spin_rho_01 = (T(Id_opt, Id_mech, s).dag())
    spin_rho_00 = (T(Id_opt, Id_mech, s) * T(Id_opt, Id_mech, s).dag())

    spin_pop = (T(Id_opt, Id_mech, s).dag() * T(Id_opt, Id_mech, s))
    phonon_pop = (T(Id_opt,b, Id_spin).dag() * T(Id_opt,b, Id_spin))
    photon_pop = (T(a, Id_mech, Id_spin).dag() * T(a, Id_mech, Id_spin))


    meas = [ca.dag() * ca, cb.dag() * cb, cb1.dag() * cb1, cs.dag() * cs, spin_pop, phonon_pop, photon_pop, spin_rho_00, spin_rho_01, spin_rho_10, spin_rho_11]

##########################################################
#     Cooling Protocol (uncomment if using this)
##########################################################
   
    me_cool = mesolve(H_cool, psi0_cool, tlist_cool, c_op_list, meas)

    # cooled_state = me_cool.states[time_scale_cool - 1]
    # cooled_spin_state_dm = cooled_state.ptrace(2)
    # cooled_phonon_state_dm = cooled_state.ptrace(1)
    # cooled_photon_state_dm = cooled_state.ptrace(0)

    plt.plot(tlist_cool, me_cool.expect[4], 'o', color = 'red', label = 'spin')
    plt.plot(tlist_cool, me_cool.expect[5], 'o', color = 'blue', label = 'phonon')
    plt.plot(tlist_cool, me_cool.expect[6], 'o', color = 'green', label = 'photon')
    plt.legend()
    plt.show()

    #Visualizing the spin state after heralding
    # xlabels_spin = ['0', '1']
    # xlabels_phonon = ['0', '1', '2']
    # xlabels_photon = ['0', '1', '2']


    # fig1, ax1 = matrix_histogram(cooled_spin_state_dm, xlabels_spin, xlabels_spin, limits=[0,1])
    # ax1.view_init(azim=-55, elev=45)

    # fig2, ax2 = matrix_histogram(cooled_phonon_state_dm, xlabels_phonon, xlabels_phonon, limits=[0,1])
    # ax2.view_init(azim=-55, elev=45)

    # # fig3, ax3 = matrix_histogram(cooled_photon_state_dm, xlabels_photon, xlabels_photon, limits=[0,1])
    # # ax3.view_init(azim=-55, elev=45)

    # plt.show()

##########################################################################
#    Monte-Carlo Simulation for heralding + swap (uncomment if using this)
##########################################################################

    # ntraj = [numb] # list of number of trajectories to avg. over

    # mc = mcsolve(H, psi0, tlist, c_op_list, [], ntraj)

    # n_first_a = 0
    # n_a_then_bad = 0
    # n_b_then_a = 0
    # n_b1_then_a = 0
    # n_other = 0
    # click = []

    # #print(mc.col_times)
    # #print(mc.col_which[1])
    # #print(mc.col_times[1])
    # #print((mc.col_times))
    # state = qt.ket2dm(psi0-psi0)

    # for traj in np.arange(numb):

    #     token = 0

    #     c_number = len(mc.col_times[traj])

    #     if (c_number > 0):

    #         for nc in np.arange(c_number):

    #             collapse_id = mc.col_which[traj][nc]

    #             if (collapse_id == 0):

    #                 if (token == 0):
    #                     n_first_a = n_first_a + 1
    #                     token = 1
                
    #                 if ((token == 2) or (token == 3)):
    #                     n_b_then_a = n_b_then_a + 1
    #                     click.append(traj)
    #                     break
                
    #             if (collapse_id == 1):

    #                 if (token == 1):
    #                     n_a_then_bad = n_a_then_bad + 1
    #                     click.append(traj)
    #                     break
                
    #                 token = 2

    #             if (collapse_id == 2):

    #                 if (token == 1):
    #                     n_a_then_bad = n_a_then_bad + 1
    #                     click.append(traj)
    #                     break
                
    #                 token = 3

    #         sum_a = sum(mc.col_which[traj])

    #         if (sum_a == 0):

    #             click.append(traj)
    #             #state = state + qt.ket2dm(mc.states[traj][time_scale-1])


    # P_first_a = n_first_a/numb
    # P_a_then_bad = n_a_then_bad/numb
    # P_b_then_a = n_b_then_a/numb

    # #state_drift = state/(n_first_a-n_a_then_bad)

    # #drift_op =  ((state_drift.ptrace(1)) * (qt.ket2dm(basis(N_mech, 1)))).tr()

    # P_herald = P_first_a + P_a_then_bad + P_b_then_a

    # P_good_traj = (P_first_a - P_a_then_bad)/P_herald

    # #Heralding_Fidelity = P_good_traj * drift_op   

    
    # #print(Fidelity)

    ####################################################################################################
    ##           Heralding protocol is done. Now we proceed with swap protocol on 'good' trajectories
    ####################################################################################################

    # time_scale_swap = 5000
    # t_up_swap = 0.25/g_sm #for pi gate between phonon and spin
    # del_t_swap = t_up_swap/time_scale_swap

    # tlist_swap = np.linspace(0,t_up_swap,time_scale_swap)
    # numb_swap = 1
    # H_swap = H_b + H_s + H_sm
    # c_op_list_swap = [cb, cb1, cs]

    



    # swap_state = qt.ket2dm(psi0-psi0)
    # herald_state = qt.ket2dm(psi0-psi0)

    # for swap_traj in click:

    #     #herald_state = herald_state + qt.ket2dm(mc.states[swap_traj][time_scale-1])

    #     mc_swap = mcsolve(H_swap, mc.states[swap_traj][time_scale-1], tlist_swap, c_op_list_swap, [], [numb_swap])

    #     swap_state =  swap_state + qt.ket2dm(mc_swap.states[0][time_scale_swap -1])

    # swap_state = swap_state/(len(click))
    # swap_spin_state_dm = swap_state.ptrace(2)
    # swap_phonon_state_dm = swap_state.ptrace(1)
    # swap_photon_state_dm = swap_state.ptrace(0)

    # herald_state = herald_state/(len(click))
    # herald_spin_state_dm = herald_state.ptrace(2)
    # herald_phonon_state_dm = herald_state.ptrace(1)
    # herald_photon_state_dm = herald_state.ptrace(0)

    # ss = (swap_spin_state_dm * (qt.ket2dm(basis(2, 1)))).tr()

    # # plt.plot(tlist_swap, mc_swap.expect[4], 'o', color = 'red')
    # # plt.show()

    # F_total = P_good_traj * ss 

    # print(P_first_a)
    # print(P_a_then_bad)
    # print(P_b_then_a)
    # print(P_herald)
    # print(P_good_traj)
    # print(F_total)

    # #Visualizing the spin state after heralding
    # xlabels_spin = ['0', '1']
    # xlabels_phonon = ['0', '1', '2']
    # xlabels_photon = ['0', '1', '2']


    # fig1, ax1 = matrix_histogram(swap_spin_state_dm, xlabels_spin, xlabels_spin, limits=[0,1])
    # ax1.view_init(azim=-55, elev=45)

    # fig2, ax2 = matrix_histogram(swap_phonon_state_dm, xlabels_phonon, xlabels_phonon, limits=[0,1])
    # ax2.view_init(azim=-55, elev=45)

    # fig3, ax3 = matrix_histogram(swap_photon_state_dm, xlabels_photon, xlabels_photon, limits=[0,1])
    # ax3.view_init(azim=-55, elev=45)

    # plt.show()


        
#######################################################################################
##      for printing analytical expressions of probabilites (uncomment if using this)
#######################################################################################

    # Prob = [P_first_a_th, P_a_then_bad_th, P_ba_th, P_b1a_th]
    # P_herald_th = Prob[0] + Prob[2] + Prob[3]
    # p_good_traj = (Prob[0] - Prob[1])/P_herald_th

    # # # print(Prob)
    # print(P_first_a_th) 
     
    # print(P_a_then_bad_th)
    # print(P_ba_th + P_b1a_th)
    # print(P_herald_th)
    # print(p_good_traj)

##################################################################################