import mph
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from scipy.optimize import minimize
from fid_qutip import *

client = mph.start()
model = client.load('SiC optomech interface 193 THz VSi-chk2.mph')

#Code for Fidelity here
#FESR: Fidelity times Entanglement Success Rate

#Blackbox function for Objective 1
def Objective(hy_d, hx_d, a_d, alpha):

    imp_x = 75.0e-9
    imp_y = 30.0e-9
    imp_z = 30.0e-9

    # g_sm_x = 20.0e-9
    # g_sm_y = 17.0e-9
    # g_sm_z = 60.0e-9

    g_sm_x = 15.0e-9
    g_sm_y = 15.0e-9
    g_sm_z = 15.0e-9

    g_cc_x = 15.0e-9
    g_cc_y = 15.0e-9
    g_cc_z = 15.0e-9

    g_sm_max = 0
    g_om_max = 0


    roi = []

    #par = np.asarray(par)
    F = 0
    var1 = str(hy_d)+'[nm]'
    var2 = str(hx_d)+'[nm]'
    var3 = str(a_d)+'[nm]'
    var4 = str(6)
    var5 = str(alpha)+'[deg]'    
    
    model.parameter('hy_d', var1)
    model.parameter('hx_d', var2)
    model.parameter('a_d', var3)
    model.parameter('n_defect', var4)
    model.parameter('alpha', var5)

    try:
        
        model.solve('3D opt')
        print("3d opt done")
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
        

        print("E- avg done")
        mm = np.argmax(np.absolute(np.array(ed_avg)))

        f_opt = np.absolute(f_opt_m[mm])
        print(f_opt)
        Q_opt = Q_opt_m[mm]
        print(type(f_opt))

        #checking if f_opt lies in the optical bandgap
        if ((f_opt > 175.28e12) and (f_opt < 203.98e12)):

            model.parameter('max_mode', str(mm+1))
            model.solve('3D mech')
            print("mech done")

            f_mech_m = np.array(model.evaluate('solid.freq', unit = 'Hz', dataset = '3D mech//Solution 3'))
            Q_mech_m = np.array(model.evaluate('solid.Q_eig', unit = '1', dataset = '3D mech//Solution 3'))

            g_mb = np.array(model.evaluate("-withsol('sol4',ewfd2.freq, setind(lambda, max_mode))/4*int_bound(((n_d^2-1)*epsilon0_const*((abs(withsol('sol4',Ex, setind(lambda, max_mode))))^2+(abs(withsol('sol4',Ey, setind(lambda, max_mode))))^2+(abs(withsol('sol4',Ez, setind(lambda, max_mode))))^2-(abs(withsol('sol4',Ex, setind(lambda, max_mode))*nX))^2-(abs(withsol('sol4',Ey, setind(lambda, max_mode))*nY))^2-(abs(withsol('sol4',Ez, setind(lambda, max_mode))*nZ))^2)-((1/n_d)^2-1)/epsilon0_const*((abs(withsol('sol4',ewfd2.Dx*nX, setind(lambda, max_mode))))^2+(abs(withsol('sol4',ewfd2.Dy*nY, setind(lambda, max_mode))))^2+(abs(withsol('sol4',ewfd2.Dz*nZ, setind(lambda, max_mode))))^2))*real(u*nX+v*nY+w*nZ))/sqrt(max_mech_all(solid.disp)^2)/withsol('sol4',ewfd2.intWe, setind(lambda, max_mode))*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))", unit ='Hz', dataset = '3D mech//Solution 3'))
            g_pe = np.array(model.evaluate("-(withsol('sol4',ewfd2.freq, setind(lambda, max_mode))*epsilon0_const*n_d^4)/4/withsol('sol4',ewfd2.intWe, setind(lambda, max_mode))*int_mech(2*real(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ey, setind(lambda, max_mode)))*(p61p*solid.eXX+p62p*solid.eYY+p66p*solid.eXY) + 2*real(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))*p55p*solid.eXZ + 2*real(withsol('sol4',ewfd2.Ey, setind(lambda, max_mode))*withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))*p44p*solid.eYZ + abs(withsol('sol4',ewfd2.Ex, setind(lambda, max_mode)))^2*(p11p*solid.eXX+p12p*solid.eYY+p13p*solid.eZZ+p16p*solid.eXY) + abs(withsol('sol4',ewfd2.Ey, setind(lambda, max_mode)))^2*(p21p*solid.eXX+p22p*solid.eYY+p23p*solid.eZZ+p26p*solid.eXY) + abs(withsol('sol4',ewfd2.Ez, setind(lambda, max_mode)))^2*(p31p*solid.eXX+p32p*solid.eYY+p33p*solid.eZZ))/sqrt(max_mech_all(solid.disp)^2)*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))", unit ='1/s', dataset = '3D mech//Solution 3'))
            g_om = g_mb + g_pe

            g_sm_val = np.array(model.evaluate(['x', 'y', 'z', '((solid.el11 - solid.el22)/3/max_mech_all(solid.disp)*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))*(0.48)*10^15[Hz])'], unit ='Hz', dataset = '3D mech//Solution 3'))        

            #g_sm_val = np.array(model.evaluate(['x', 'y', 'z', '((-solid.el11 - solid.el22 + 2*solid.el33 + 2*(solid.el12 + solid.el12) - (solid.el13 + solid.el13) - (solid.el23 + solid.el23))/3/max_mech_all(solid.disp)*sqrt(hbar/(2*(1/max_mech_all((solid.disp)^2)*int_mech_all(root.material.rho*(solid.disp)^2))*real(solid.omega)))*10^15[Hz])'], unit ='Hz', dataset = '3D mech//Solution 3'))        


            x = g_sm_val[0][0]
            y = g_sm_val[1][0]
            z = g_sm_val[2][0]
            g_sm = g_sm_val[3]
            print("g eval")

            
            g_sm_avg = []

            #for picking breathing mechanical mode

            for index in np.arange(0,len(g_om)):
                roi_gsm = []
                n_points_gsm = 0
                sum_gsm = 0
                
                for b in np.arange(0, len(g_sm_val[0][0])):
                                            #50e-9                            #23e-9
                    if ((np.absolute(x[b]) < g_sm_x) and (np.absolute(y[b]) < g_sm_y) and ((np.absolute(z[b]-1.25e-7) < g_sm_z)) and ~np.isnan(g_sm[index][b])):
                                
                        n_points_gsm = n_points_gsm + 1
                        roi_gsm.append([x[b], y[b], z[b]])
                        sum_gsm = sum_gsm + g_sm[index][b]

                g_sm_avg.append(np.absolute(sum_gsm/n_points_gsm))

            print("ch1")

            max_mm1 = np.argmax(np.absolute(g_sm_avg))
            max_mm2 = np.argmax(np.absolute(g_om))
            f_mech1 = np.absolute(f_mech_m[max_mm1])
            f_mech2 = np.absolute(f_mech_m[max_mm2])

            print(f_mech1)
            print(f_mech2)

            Q_mech = Q_mech_m[max_mm2]
            g_om_max = np.absolute(g_om[max_mm2])
            g_sm_max = np.absolute(g_sm_avg[max_mm2])
            g_sm_cc = 0
            n_points_cc = 0
            #print(f_mech)

            #for averaging g_sm at the color center's position

            # for b in np.arange(0, len(g_sm_val[0][0])):
                            
            #     if ((np.absolute(x[b]) < g_cc_x) and (np.absolute(y[b]) < g_cc_y) and ((np.absolute(z[b]-1.25e-7) < g_cc_z)) and ~np.isnan(g_sm[max_mm2][b])):
                            
            #         n_points_cc = n_points_cc + 1
            #         g_sm_cc = g_sm_cc + g_sm[max_mm2][b]

            # print(n_points_cc)
            # g_sm_cc = np.absolute(g_sm_cc/n_points_cc)
            #print(n_points_gsm)
            #print(g_sm_max)

    except:
        print("some error!")
        pass
    
    
    return (g_sm_max, g_om_max, f_mech1, f_mech2, f_opt, max_mm1)

            
GSM = []
GOM = []
FM = []
FO = []
Mode = []
#h_list = np.array([325, 330, 335, 340, 345, 350, 355, 360, 370, 375, 380, 385])
hx_list = np.array([275])
hy_list = np.array([355])
a_list = np.array([365])

for hy in hy_list:

    for hx in hx_list:

        for a_d in a_list:
            
            gsm, gom, fm1, fm2, fo, m = Objective(hy, hx, a_d, 90)

            print('{0:3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(hy, hx, a_d, fo, fm1, fm2))


# print(GSM)
# print(GOM)
# print(FM)
# print(FO)
# print(Mode)

# plt.plot(h_list, np.array(GSM), 'o', color = 'red', label = 'g_sm')
# plt.plot(h_list, np.array(GOM), 'o', color = 'blue', label = 'g_om')
# plt.xlabel("h [nm]")
# plt.legend()
# #plt.ylabel("<g_sm> (Hz)")
# plt.show()







