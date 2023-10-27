import mph
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize
from fid_qutip import *

client = mph.start()
model = client.load('SiC optomech interface 193 THz VSi.mph')

#Code for Fidelity here
#FESR: Fidelity times Entanglement Success Rate

def FESR(Q_opt, Q_mech, f_opt, f_mech, g_om, g_sm):
    ## qutip code block here
    n_pump = 1000
    temp_bath = 4 #K
    T1e = 100e-6 #sec
    N=3
    times = np.linspace(0,5e-6,int(1e4))
    Delta_p=0
    Delta_e=0

    f = F_in(n_pump, temp_bath, f_mech, Q_mech, f_opt, Q_opt, g_om)
    r = succ_rate(n_pump, temp_bath, f_mech, Q_mech, f_opt, Q_opt, g_om)

    F_total = rabi_fidelity(
        times,
        F_in = f,
        N=N,
        f0=f_mech,
        Qmech=Q_mech,
        T1e=T1e,
        g_sm=g_sm,
        Temp=temp_bath,
        Delta_p=Delta_p,
        Delta_e=Delta_e,
        plotting=False
    )
    return (F_total*r, F_total, r)

#Blackbox function for Objective 1
def Objective(hy_d, hx_d, a_d):

    imp_x = 75.0e-9
    imp_y = 30.0e-9
    imp_z = 30.0e-9

    g_sm_x = 15.0e-9
    g_sm_y = 15.0e-9
    g_sm_z = 15.0e-9


    roi = []

    #par = np.asarray(par)
    F = 0
    var1 = str(hy_d)+'[nm]'
    var2 = str(hx_d)+'[nm]'
    var3 = str(a_d)+'[nm]'
    var4 = str(6)    
    
    model.parameter('hy_d', var1)
    model.parameter('hx_d', var2)
    model.parameter('a_d', var3)
    model.parameter('n_defect', var4)

    try:
        
        model.solve('3D opt')
        f_opt_m = np.array(model.evaluate('ewfd2.freq', unit='Hz', dataset='3D opt//Solution 4'))
        Q_opt_m = np.array(model.evaluate('ewfd2.Qfactor', unit='Hz', dataset='3D opt//Solution 4'))
        Exyz = np.array(model.evaluate(['x', 'y', 'z', '0.5*ewfd2.normE*ewfd2.normE*epsilon0_const + 0.5*ewfd2.normB*ewfd2.normB/mu0_const'], unit = 'J/m^3', dataset = '3D opt//Solution 4'))

        lx = Exyz[0][0]
        ly = Exyz[1][0]
        lz = Exyz[2][0]
        ed = Exyz[3]

        ed_avg = []

        for e_mode in np.arange(0, len(f_opt_m)):

            roi = []
            n_points = 0
            sum = 0
                
            for b in np.arange(0, len(lx)):
                    
                if ((np.absolute(lx[b]) < imp_x) and (np.absolute(ly[b]) < imp_y) and ((np.absolute(lz[b]-1.25e-7) < imp_z))):
                        
                    n_points = n_points + 1
                    roi.append([lx[b], ly[b], lz[b]])
                    sum = sum + ed[e_mode][b]


            ed_avg.append(sum/n_points)

        
        mm = np.argmax(np.absolute(np.array(ed_avg)))

        f_opt = np.absolute(f_opt_m[mm])
        Q_opt = Q_opt_m[mm]

        #print(f_opt)
        #print(type(f_opt))

        #checking if f_opt lies in the optical bandgap
        if ((f_opt > 175.28e12) and (f_opt < 203.98e12)):
            
            model.parameter('max_mode', str(mm+1))
            model.solve('3D mech')

            f_mech_m = np.array(model.evaluate('solid.freq', unit = 'Hz', dataset = '3D mech//Solution 3'))
            Q_mech_m = np.array(model.evaluate('solid.Q_eig', unit = '1', dataset = '3D mech//Solution 3'))

            g_mb = np.array(model.evaluate("-withsol('sol4',ewfd2.freq, setind(lambda, max_mode))/4*int_bound(((n_d^2-1)*epsilon0_const*((abs(withsol('sol4',Ex, setind(lambda, max_mode))))^2+(abs(withsol('sol4',Ey, setind(lambda, max_mode))))^2+(abs(withsol('sol4',Ez, setind(lambda, max_mode))))^2-(abs(withsol('sol4',Ex, setind(lambda, max_mode))*nX))^2-(abs(withsol('sol4',Ey, setind(lambda, max_mode))*nY))^2-(abs(withsol('sol4',Ez, setind(lambda, max_mode))*nZ))^2)-((1/n_d)^2-1)/epsilon0_const*((abs(withsol('sol4',ewfd2.Dx*nX, setind(lambda, max_mode))))^2+(abs(withsol('sol4',ewfd2.Dy*nY, setind(lambda, max_mode))))^2+(abs(withsol('sol4',ewfd2.Dz*nZ, setind(lambda, max_mode))))^2))*real(u*nX+v*nY+w*nZ))/sqrt(max_mech_all(solid.disp)^2)/withsol('sol4',ewfd2.intWe, setind(lambda, max_mode))*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))", unit ='Hz', dataset = '3D mech//Solution 3'))
            g_pe = np.array(model.evaluate("-(withsol('sol4',ewfd2.freq, setind(lambda, max_mode))*epsilon0_const*n_d^4)/4/withsol('sol4',ewfd2.intWe, setind(lambda, max_mode))*int_mech(2*real(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ey, setind(lambda, max_mode)))*(p61p*solid.eXX+p62p*solid.eYY+p66p*solid.eXY) + 2*real(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))*p55p*solid.eXZ + 2*real(withsol('sol4',ewfd2.Ey, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))*p44p*solid.eYZ + abs(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode)))^2*(p11p*solid.eXX+p12p*solid.eYY+p13p*solid.eZZ+p16p*solid.eXY) + abs(withsol('sol4',ewfd2.Ey, setind(lambda, max_mode)))^2*(p21p*solid.eXX+p22p*solid.eYY+p23p*solid.eZZ+p26p*solid.eXY) + abs(withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))^2*(p31p*solid.eXX+p32p*solid.eYY+p33p*solid.eZZ))/sqrt(max_mech_all(solid.disp)^2)*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))", unit ='1/s', dataset = '3D mech//Solution 3'))
            g_om = g_mb + g_pe

            g_sm_val = np.array(model.evaluate(['x', 'y', 'z', '((solid.el11 - solid.el22)/3/max_mech_all(solid.disp)*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))*(0.48)*10^15[Hz])'], unit ='Hz', dataset = '3D mech//Solution 3'))        

            x = g_sm_val[0][0]
            y = g_sm_val[1][0]
            z = g_sm_val[2][0]
            g_sm = g_sm_val[3]

            
            g_sm_avg = []

            for index in np.arange(0,len(g_om)):
                roi_gsm = []
                n_points_gsm = 0
                sum_gsm = 0
                
                for b in np.arange(0, len(g_sm_val[0][0])):
                            
                    if ((np.absolute(x[b]) < g_sm_x) and (np.absolute(y[b]) < g_sm_y) and ((np.absolute(z[b]-1.25e-7) < g_sm_z))):
                                
                        n_points_gsm = n_points_gsm + 1
                        roi_gsm.append([x[b], y[b], z[b]])
                        sum_gsm = sum_gsm + g_sm[index][b]


                g_sm_avg.append(np.absolute(sum_gsm/n_points_gsm))

            max_mm = np.argmax(np.absolute(g_sm_avg))
            f_mech = np.absolute(f_mech_m[max_mm])
            Q_mech = Q_mech_m[max_mm]
            g_om_max = np.absolute(g_om[max_mm])
            g_sm_max = np.absolute(g_sm_avg[max_mm])

            #print(f_mech)
            #print(type(f_mech))
            #print(g_om_max)
            #print(type(g_om_max))

            #checking if f_mech lies in the mechanical bandgap
            if ((f_mech > 4.96e9) and (f_mech < 7.37e9)):

                F, f_total, rate = fom(1000, 4, Q_opt, Q_mech, f_opt, f_mech, g_om_max, g_sm_max)
                print([F, f_total, rate])

                if (f_total > 0.0):
                    # display information
                    #print ('{0:3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}   {9: 3.6f}   {10: 3.6f}   {11: 3.6f}'.format(par[0], par[1], par[2], g_om_max, g_sm_max, f_opt, f_mech, Q_opt, Q_mech, f_total, rate, F))
                    with open("output_SiC.txt", "a") as mf:
                        print(('{0:3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}   {9: 3.6f}   {10: 3.6f}   {11: 3.6f}'.format(hy_d, hx_d, a_d, g_om_max, g_sm_max, f_opt, f_mech, Q_opt, Q_mech, f_total, rate, F)), file=mf)

                else:
                    F = 0
                    #print('total fidelity less than or equal to 0.95, skipping to next iteration')
                    with open("output_SiC.txt", "a") as mf:
                        print('total fidelity less than or equal to 0.95, skipping to next iteration', file=mf)                                       

            else:
                #print('f_mech out of mechanical bandgap, skipping to next iteration')
                with open("output_SiC.txt", "a") as mf:
                    print('f_mech out of mechanical bandgap, skipping to next iteration', file=mf)


        else:
            #print('f_opt out of optical bandgap, skipping to next iteration')
            with open("output_SiC.txt", "a") as mf:
                    print('f_opt out of optical bandgap, skipping to next iteration', file=mf)



    except:
        print("Bad news: there is some error!!!")
        pass

    #print ('{0:3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}   {9: 3.6f}   {10: 3.6f}   {11: 3.6f}'.format(par[0], par[1], par[2], g_om_max, g_sm_max, f_opt, f_mech, Q_opt, Q_mech, f_total, rate, F))
    return F

with open("output_SiC.txt", "a") as mf:
    print  ('{0:9s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}   {5:9s}   {6:9s}   {7:9s}   {8:9s}   {9:9s}   {10:9s}   {11:9s}'.format('hy_d(nm)', 'hx_d(nm)', 'a_d(nm)', 'g_om(Hz)', 'g_sm(Hz)', 'f_opt(Hz)', 'f_mech(Hz)', 'Q_opt', 'Q_mech', 'F_total', 'S.Rate(Hz)', 'FESR(Hz)'), file=mf)

# Bounded region of parameter space
pbounds = {'hy_d': (340, 370), 
           'hx_d': (270, 300), 
           'a_d': (380, 410)
            }

#Calling Optimizer
optimizer = BayesianOptimization(
    f=Objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=20,
    n_iter=100,
)

print(optimizer.max)






