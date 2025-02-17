# issues
# are V_I and V_G supposed to multiplied by BW?
# are V_I ad V_G supposed to be fractional (0.12/0.16) or x10 (120, 160)
# what's going on with k_a1/k_a2/k_a3
# what is ka?
# glucose units: mg/dL or 

# matlab simulator has glucagon boluses

# closed form for gut absorption model???


# modifications: V_I, V_G L-> mL/kg
# F_01, EGP_0 -> mmol/min to umol (kg min)
# 

import numpy as np
from scipy.optimize import fsolve

class VariableParameter:
    def __init__(self, target=0, val=0):
        self.target = target
        self.val = val

class Variability:
    def __init__(self):
        self.ka1 = VariableParameter()
        self.ka2 = VariableParameter()
        self.ka3 = VariableParameter()
        self.ka = VariableParameter()
        self.k12 = VariableParameter()
        self.F01 = VariableParameter()

class HovorkaModel:
    def __init__(self):
        self.w = 74.9
        self.MCHO = 180.1577
        self.EGP0 = 16.1               # EGP extrapolated to zero insulin concentration [mmol/min]
        self.F01 = 9.7                 # Non-insulin-dependent glucose flux [mmol/min]
        self.k12 = 0.066               # Transfer rate [min]
        self.RTh = 14
        self.RCl = 0.003
        self.St = 51.2e-4              # Insulin sensitivity of distribution/transport [L/min*mU]
        self.Sd = 8.2e-4               # Insulin sensitivity of disposal [L/min*mU]
        self.Se = 520e-4               # Insluin sensitivity of EGP [L/mU]
        self.k_a1 = 0.006              # Deactivation rate of insulin on distribution/transport [1/min]
        self.k_a2 = 0.06               # Deactivation rate of insulin on dsiposal [1/min]
        self.k_a3 = 0.03               # Deactivation rate of insulin on EGP [1/min]
        self.ke = 0.138                # Insulin elimination from Plasma [1/min]
        self.Vi = 120                  # Insulin volume [L]
        self.Vg = 160                  # Glucose volume [L]
        self.tau_G = 40                # Time-to-maximum CHO absorption [min]
        self.tau_I = 55                # Time-to-maximum of absorption of s.c. injected short-acting insulin [min]
        self.Bio = 0.8                 # CHO bioavailability [1]

        self.eInsSub1 = 1
        self.eInsSub2 = 2
        self.eInsPlas = 3
        self.eInsActT = 4
        self.eInsActD = 5
        self.eInsActE = 6
        self.eGluPlas = 7
        self.eGluComp = 8
        self.eGluInte = 9
        self.eGutAbs = 10
        self.eGluMeas = 11
        
        self.variability = Variability()
        
        
        



    def calc_TDD():
        w = 74.9
        MCHO = 180.1577


        EGP0 = 16.1               # EGP extrapolated to zero insulin concentration [mmol/min]
        F01 = 9.7                 # Non-insulin-dependent glucose flux [mmol/min]
        k12 = 0.066               # Transfer rate [min]
        RTh = 14
        RCl = 0.003

        St = 51.2e-4             # Insulin sensitivity of distribution/transport [L/min*mU]
        Sd = 8.2e-4              # Insulin sensitivity of disposal [L/min*mU]
        Se = 520e-4              # Insluin sensitivity of EGP [L/mU]
        k_a1 = 0.006               # Deactivation rate of insulin on distribution/transport [1/min]
        k_a2 = 0.06                # Deactivation rate of insulin on dsiposal [1/min]
        k_a3 = 0.03                # Deactivation rate of insulin on EGP [1/min]
        ke = 0.138                # Insulin elimination from Plasma [1/min]

        Vi = 120                  # Insulin volume [L]
        Vg = 160                  # Glucose volume [L]

        tau_G = 40                 # Time-to-maximum CHO absorption [min]
        tau_I = 55                 # Time-to-maximum of absorption of s.c. injected short-acting insulin [min]

        Bio = 0.8                  # CHO bioavailability [1]

        Gs0 = 6.5


        Q10 = Gs0 * Vg
        Fn = Q10 * (F01 / 0.85) / (Q10 + Vg)
        Fr = RCl * (Q10 - RTh * Vg) * (Q10 > RTh * Vg)
        Slin = np.roots([(-Q10 * St * Sd - EGP0 * Sd * Se), (-k12 * EGP0 * Se + (EGP0 - Fr - Fn) * Sd), k12 * (EGP0 - Fn - Fr)])

        f = lambda x: -Fn -Q10*St*x +k12*(Q10 * St * x)/(k12 + Sd * x) -Fr+EGP0*np.exp(-Se*x)

        S = fsolve(f, max(Slin))
        
        Ip0 = S[0]
        
        Ub = 60 * Ip0 * ke / (1e6 / (Vi * w))
            
        basalGlucose = 6.5
        
        carbF = min(max(round(2*(MCHO * (0.4 * max(St, 16e-4) + 0.6 * min(max(Sd, 3e-4), 12e-4)) * basalGlucose * Vg)/(ke * Vi))/2, 2), 25)
        
        print(carbF)
        
        TDD = min(max(round(Ub*24+200/carbF, 2), 10), 110)
        return TDD

    def model(self, t, y, u):
        dydt = np.zeros(y.shape)
        
        # Subcutaneous insulin absorption subsystem (U).
        dydt[self.eInsSub1] = u / 60 - y[self.eInsSub1] * self.variability.ka.val
        dydt[self.eInsSub2] = y[self.eInsSub1] * self.variability.ka.val - y[self.eInsSub2] * self.variability.ka.val
        
        # Subcutaneous glucagon boluses absorption subsystem.       
        # Plasma insulin kinetics subsystem (mU/L).
        dydt[self.eInsPlas] = 1e6 * y[self.eInsSub2] * self.variability.ka.val / (self.Vi * self.w) - y[self.eInsPlas] * self.variability.ke.val
        
        # Plasma insulin action subsystem ((1/min) / (1/min) / no-units).
        dydt[self.eInsActT] = -self.variability.ka1.val * y[self.eInsActT] + self.variability.ka1.val * self.variability.St.val * y[self.eInsPlas]
        dydt[self.eInsActD] = -self.variability.ka2.val * y[self.eInsActD] + self.variability.ka2.val * self.variability.Sd.val * y[self.eInsPlas]
        dydt[self.eInsActE] = -self.variability.ka3.val * y[self.eInsActE] + self.variability.ka3.val * self.variability.Se.val * y[self.eInsPlas]
        
        # Gut absorption subsystem (umol/kg/min).
        Um = 0
        for m = 1:length(self.meals)
            Um = Um + self.meals(m).gutAbsorptionModel(t, self.meals(m));
        end
        
        # Glucose kinetics subsystem (umol/kg).
        dydt[self.eGluPlas] = -((self.variability.F01.val / 0.85) / (y[self.eGluPlas] + self.Vg) + y[self.eInsActT] * y[self.eGluPlas] 
            +self.variability.k12.val * y[self.eGluComp] 
            -self.RCl * (y[self.eGluPlas] - self.RTh * self.Vg) * (y[self.eGluPlas] > self.RTh * self.Vg) 
            +self.variability.EGP0.val * (exp(-y[self.eInsActE]) + exp(-1/(GluPlas * self.TGlu))) 
            +Um;
        
        dydt[self.eGluComp] = y[self.eInsActT] * y[self.eGluPlas] 
            -(self.variability.k12.val + y[self.eInsActD]) * y[self.eGluComp];
        
        # Glucose sensor (mmol/L).
        dydt[self.eGluInte] = (y[self.eGluPlas] / self.Vg - y[self.eGluInte]) / self.TauS;

if __name__ == '__main__':
    TDD = calc_TDD()
    print(TDD)