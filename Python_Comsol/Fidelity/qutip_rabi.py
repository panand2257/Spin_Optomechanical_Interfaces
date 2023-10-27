import qutip
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sp_opt
T = tensor

#from tqdm.notebook import tqdm
############# ------------------- ####################
# everything is measured in units of 1 Hz and 1 s and Kelvin

F0 = 4.75e9#4.7578e9 # 2.7 GHz omega
Fopt = 193e14
k_per_h = 20836617636.1328 # Hz per K
n_per_T_mech = k_per_h/F0 # photons per K
n_per_T_opt = k_per_h/Fopt
Delta_p = 0
Delta_sig = 0
#hbar = 6.626*10^(-34)/2/numpy.pi

# everything is measured in units of 1MHz and 1us and Kelvin

F0 = 4.7578e9/2/np.pi # 2.7 GHz omega
k_per_h = 20.84e9 # MHz per K
n_per_T = k_per_h/F0 # photons per K
Delta_p = 10e6
Delta_sc = 500e6
Delta_e = 500e6

def prep(
    N=3,
    f0=5e9,
    Qmech=1e5,
    T1e=1000e6,
    g_sm=1e6,
    Temp=0.015,
    Delta_p=0,
    Delta_e=0
):
    ap = destroy(N)                #annihilation operator (phonon cavity)
    apd = ap.dag()                  #creation operator (phonon cavity)
    se = destroy(2)                #annihilation operator (spin state)
    sed = se.dag()                  #creation operator (spin state)
    idN = identity(N)             #identity operator (bosonic modes)
    id2 = identity(2)             #identity operator (2-level)

    nP = num(N)                   #number operator (phonon cavity)
    Np = T(nP,id2)            #number operator (phonon cavity, full Hilbert space matrix)
    ne = num(2)                   #number operator (electron spin)
    Ne = T(idN,ne)            #number operator (electron spin, full Hilbert space matrix)

    Ap = T(ap,id2)              #annihilation operator (phonon cavity, full Hilbert space matrix)
    Se = T(idN,se)              #annihilation operator (electron spin, full Hilbert matrix)

    H_sm = 2*np.pi*g_sm * T(ap,sed)     #spin-mechanical interaction Hamiltonian
    H_sm += H_sm.dag()            #...plus Hermitian conjugate

    ce = (1/T1e)**0.5 * T(idN,se)  # decay-adjusted annihilation operator (electron spin)
    
    nT = n_per_T*Temp                               # thermal photon count
    gamma = 4*np.pi*f0/Qmech                     # phonon decay rate
    cp = (gamma/2*(nT+1))**0.5 * T(ap,id2)      # decay-adjusted phonon decay rate
    cp1 = (gamma/2*nT)**0.5 * T(apd,id2)

    #gammaA = 1/T1sc                                # microwave photon decay rate
    #csc = (gammaA/2*(nT+1))**0.5 * T(ssc, idN, id2)# decay-adjusted microwave photon decay rate
    
    H_im = - 0.5j * (cp.dag()*cp + cp1.dag()*cp1 + ce.dag()*ce)
    
    H_p = 2*np.pi*Delta_p * T(apd*ap, id2)
    H_e = 2*np.pi*Delta_e * T(idN, sed*se)

    vac = T(fock(N,0), fock(2,0))
    vacE = T(fock(N,0), fock(2,1))
    
    #proj_vac = T(projection(2,0,0), idN, id2)
    #proj_notvac = T(id2, idN, id2) - proj_vac
    
    return Ap,Se,Np,Ne,cp,cp1,ce,H_sm,H_im,H_p,H_e,vac,vacE,nT,gamma#,proj_vac,proj_notvac

def run(
        times,
        F_in,
        N=3,
        f0=5e9,
        Qmech=1e5,
        T1e=1000e-6,
        g_sm=1e6,
        Temp=0.015,
        Delta_p=0,
        Delta_e=0
):
    Ap,Se,Np,Ne,cp,cp1,ce,H_sm,H_im,H_p,H_e,vac,vacE,nT,gamma = \
    prep(
        N=N,
        f0=f0,
        Qmech=Qmech,
        T1e=T1e,
        g_sm=g_sm,
        Temp=Temp,
        Delta_p=Delta_p,
        Delta_e=Delta_e
    )
    
    me_results = mesolve(
        H_sm + H_e + H_p + H_im,
        F_in*T(fock(N,1),fock(2,0)),
        times,
        options=Options(normalize_output=False,store_states=True,
                            nsteps=5000,atol=1e-12,rtol=1e-12),
        progress_bar=None,
        c_ops=[ce,cp,cp1]
    )

    return me_results


def rabi_fidelity(
    times,
    F_in,
    N,
    f0,
    Qmech,
    T1e,
    g_sm,
    Temp,
    Delta_p,
    Delta_e,
    plotting=False,
):
    
    Ap,Se,Np,Ne,cp,cp1,ce,H_sm,H_im,H_p,H_e,vac,vacE,nT,gamma = \
        prep(N=N, g_sm=g_sm, T1e=T1e, Qmech=Qmech, Temp=Temp)

    me_results = run(
        times,
        F_in=F_in,
        N=N,
        f0=f0,
        Qmech=Qmech,
        T1e=T1e,
        g_sm=g_sm,
        Temp=Temp,
        Delta_p=Delta_p,
        Delta_e=Delta_e
    )

    norms = [s.norm()**2 for s in me_results.states]
    cp_exp  = [expect(cp.dag() *cp,s )/s.norm()**2 for s in me_results.states]
    cp1_exp = [expect(cp1.dag()*cp1,s)/s.norm()**2 for s in me_results.states]
    ce_exp  = [expect(ce.dag() *ce,s )/s.norm()**2 for s in me_results.states]

    ap_exp = [expect(Ap.dag()*Ap,s) for s in me_results.states]
    se_exp = [expect(Se.dag()*Se,s) for s in me_results.states]

    if plotting:
        plt.figure()
        plt.semilogy(times,se_exp)
        plt.ylim([1e-1,1e0])
        plt.plot(times,ap_exp)
        plt.plot(times,np.linspace(1/np.exp(1),1/np.exp(1),np.size(times)))
        plt.plot(times,np.linspace(0.9,0.9,np.size(times)))
        plt.plot(times,np.linspace(0.8,0.8,np.size(times)))
        plt.xlabel('time [s]')
        plt.ylabel('Expectation value [norm 1]')

    return np.max(se_exp)


