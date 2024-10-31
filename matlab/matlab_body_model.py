import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from bisect import bisect_right

def maxfunc(x, a, ep):
    return (x+np.sqrt(x**2+ep))/2

def hill(t,tau,A,k):
    return A/(1+np.exp(-k*(t-tau)))

def delta(t,tau,A,k):
    Hval = A/(1+np.exp(-k*(t-tau)))
    return k*Hval*(1-Hval/A)
# Glucose subsystem
Vg=1.49; k1=0.065; k2=0.079
# Insuline subsystem
VI=0.04; m1=0.379; m2=0.673; m4=0.269; m5=0.0526; m6=0.8118; HEb=0.112
# Endogenous glucose production
kp1= 2.7; kp2=0.0021; kp3=0.009
kp4=0.0786; ki=0.0066
# Glucose rate of appearance
kmax=0.0465; kmin=0.0076; kabs=0.023; kgri=0.0465; f=0.9; a=0.00016; b=0.68; c=0.00023; d=0.009
# Glucose utilization
Fcns=1; Vm0=4.65; Vmx=0.034; Km0 = 471.13
# Km0=466.21; 
P2u=0.084;       
# Fcns=1; Vm0=2.5; Vmx=0.047; Km0=225.59; P2u=0.0331;

# Insuline secretion
K=0.99; Alpha=0.013; Beta=0.05; Gamma=0.5
# Renal excretion
ke1=0.0007; ke2=269
# Subcatenuous Insuln
kd = 0.0164; ka1 = 0.0018; ka2 = 0.0182
# Glucagon system
Delta = 0.682; Sigma = 1.093; n = 0.15
Zeta = 0.009; Rho = 0.57; kH = 0.16; Hb = 93
# Subcatenuous Glucagon
kh1 = 0.0164; kh2 = 0.0018; kh3 = 0.0182

# want to make the D delta function integrate to 1
D_delta_correction = 1 / 0.982013790038

def basal_states(Gb):
    Sb= 0; # (m6-HEb)/m5; 
    IIRb = 0
    m3=HEb*m1/(1-HEb)
    Ipb= IIRb/(m2+m4-(m1*m2)/(m1+m3)) # 2/5*Sb*(1-HEb)/m4
    Ilb= Ipb*(m2/(m1+m3))
    Ib=Ipb/VI
    Ipob= 0;# Sb/Gamma
    EGPb= 2.4;# (1/(2*(k1+kp2)))*(Fcns*k1+k1*kp1+2*Fcns*kp2+k2*Km0*kp2-Ib*k1*kp3-Ipob*k1*kp4+k1*Vm0+kp2*Vm0...
        #-sqrt(-4*k1*k2*Km0*kp2*(Fcns-kp1+Ib*kp3+Ipob*kp4)+(k2*Km0*kp2+k1*(Fcns-kp1+Ib*kp3+Ipob*kp4)+(k1+kp2)*Vm0)^2)); 
    # two possible solutions! but the (+) solution give negative glucose basal values! 
    # Gb = 160;%72.763;
    Gpb= Gb*Vg; # (kp1-EGPb-kp3*Ib)/kp2;
    Gtb=(Fcns-EGPb+k1*Gpb)/k2
    Isc1ss = IIRb/(kd+ka1)
    Isc2ss = kd*Isc1ss/ka2
    kp1 = EGPb+kp2*Gpb+kp3*Ib
    Km0 = (Vm0*Gtb)/(EGPb-Fcns) -Gtb

    SRHb = n*Hb
    Gth = Gb
    SRsHb = max(Sigma*(Gth-Gb)+SRHb,0)
    XHb = 0
    Ith = Ib
    # Subcutaeneous Glucagon
    IGRb = 0
    Hsc1ss = IGRb/(kh1+kh2)
    Hsc2ss = kh1*Hsc1ss/kh3
    return (Gb,Gpb,Gtb,Ilb,Ipb,Ipob,Ib,IIRb,Isc1ss,Isc2ss,kp1,Km0,Hb,SRHb,Gth,SRsHb, XHb,Ith,IGRb,Hsc1ss,Hsc2ss)

# function xdot=DallaMan2018f(t,x,type,BW,D,uI,uG,t0,Gb,EGPb,IIRb,IGRb)
# t: time
# x: state vector
# type: 'diabetic' or 'normal', not relevant here
# BW: body weight
# Gb: basal glucose
# D: carbohydrates
# uI: insulin infusion
# uG: rate of appearance of glucose??

# carb_doses is an array of (time, dose) tuples sorted by time
# since the only place we are using t is in the carbs delta function, we can compute t relative to the last dose
# TODO: this could mess with kempt though; we need to see how to handle this
def model(t,x,type,BW,Gb,carb_doses,uI,uG,t0):
    diractime=1e-3; # time to approximate the dirac delta
    [_,_,_,_,_,_,Ib,IIRb,_,_,kp1,Km0,Hb,SRHb,Gth,_, _, Ith,IGRb,_,_]=basal_states(Gb)
    D = 1e-9
    for i in range(len(carb_doses) - 1, -1, -1):
        if t >= carb_doses[i][1] and carb_doses[i][0] > 0:
            t = t - carb_doses[i][1]
            D = carb_doses[i][0]
            break

    # [Ib,IIRb,kp1,Km0,Hb,SRHb,Gth,Ith,IGRb]
    # print(carb_doses)
    # define the state variables and all the (state dependent) functions
    # Glucose subsystem
    Gp=x[1];Gt=x[2]; G=Gp/Vg; 
    # Insuline subsystem
    Il=x[3]; Ip=x[4]; I=Ip/VI 
    # Endogenous glucose production
    I1=x[5]; Id=x[6]; 
    # Glucose rate of appearance
    Qsto1=x[7]; Qsto2=x[8]; Qgut=x[9]; Ra=f*kabs*Qgut/BW
    # Glucose utilization
    X=x[10]; Uii=Fcns; Uid=(Vm0+Vmx*X)*Gt/(Km0+Gt)
    # Insuline secretion
    # Ipo=x(11); Y=x(12); S= 0;%Gamma*Ipo;
    # Glucagon kinetics
    SRsH = x[11]; H = x[12]; XH = x[13]
    # Endogeneous glucagon production
    EGP= kp1-kp2*Gp-kp3*Id + Zeta* XH;# eq10
    # Glucose renal excretion
    E=ke1*(Gp-ke2)*hill(Gp,ke2,1,4)
    # subcutaneous insulin infusion
    Isc1 = x[14]; Isc2 = x[15]
    Rai = ka1*Isc1 + ka2*Isc2; # pmole/(kg*min)or  pmole/kg/min
    # Subcataneous Glucagon
    Hsc1 = x[16]; Hsc2 = x[17]
    Rah = kh3 * Hsc2
    # define the equations
    xdot=np.zeros(len(x))
    # Glucose subsystem
    xdot[1]=EGP+Ra-Uii-E-k1*Gp+k2*Gt
    xdot[2]=-Uid+k1*Gp-k2*Gt
    Gdot=xdot[1]/Vg
    # Insulin subsystem
    HE = HEb; # -m5*S+m6
    m3=HE*m1/(1-HE)
    xdot[3]= -(m1+m3)*Il+m2*Ip
    xdot[4]= -(m2+m4)*Ip+m1*Il+Rai
    # Endogenous glucose production
    xdot[5]=-ki*(I1-I)
    xdot[6]=-ki*(Id-I)
    # Glucose rate of appearance
    Qsto=Qsto1+Qsto2
    
    # kempt is rate of gastric emptying, is a "nonlinear function of Qsto"
    # gastric emptying is movement of the food from stomach into small intestine
    kempt = kmin+(kmax-kmin)/2*(np.tanh(5/(2*D*(1-b))*(Qsto-b*D))-np.tanh(5/(2*D*c)*(Qsto-c*D))+2); 
    
    # come up with a better way to model this
    xdot[7]=-kgri*Qsto1 + D_delta_correction * D*delta(t,1,1,4);# 1/diractime*D*((t-t0)<diractime)
    xdot[8]=-kempt*Qsto2+kgri*Qsto1
    xdot[9]=-kabs*Qgut+kempt * Qsto2
    # Glucose utilization
    xdot[10]=-P2u*(X-I+Ib)
    # Insuline secretion
    #h=Gpb/Vg;
    #Spo=Y+K*Gdot*(Gdot>0)+Sb;
    #xdot(11)=-Gamma*Ipo+Spo;
    #xdot(12)=-Alpha*(Y+Sb-(Sb+Beta*(G-h))*(Beta*(h-G)<Sb));
    # Glucagon Subsystem
    SRdH = Delta*maxfunc(-Gdot,0,0.0001)
    SRH = SRsH+SRdH
    xdot[11] = -Rho*(SRsH -  maxfunc((Sigma*(Gth-G)/(maxfunc(I-Ith,0,0.0001)+1))+SRHb,0,0.0001))
    xdot[12] = -n*H +SRH +Rah
    xdot[13] = -kH*XH +kH* maxfunc(H-Hb,0,0.0001)
    # Subcatenous Insulin
    xdot[14] = -(kd+ka1)*Isc1 +IIRb 
    # + (1/78)*uI*6944.4*delta(t,1000,1,4);# +uI*(t<30)

    # load('uu.mat');
    # 
    # aaa = 20.9*delta(t,30,1,4);
    # uu = [uu;[t aaa]];150
    # save('uu.mat','uu');

    xdot[15] = kd*Isc1 - ka2*Isc2
    # Subcatenous Glucagon
    xdot[16] = -(kh1+kh2)*Hsc1  
    # + (1/78)*uG*1e6*delta(t,150,1,4)
    xdot[17] = kh1*Hsc1 - kh3*Hsc2
    return xdot


if __name__ == '__main__':
    BW=78 # body parameter
    Gb = 130 # body parameter
    (Gb,Gpb,Gtb,Ilb,Ipb,Ipob,Ib,IIRb,Isc1ss,Isc2ss,kp1,Km0,Hb,SRHb,Gth,SRsHb, XHb,Ith,IGRb,Hsc1ss,Hsc2ss) = basal_states(Gb)
    x0=[0, Gpb,Gtb,Ilb,Ipb,Ib,Ib,0,0,0,0,SRsHb,Hb,XHb,Isc1ss,Isc2ss,Hsc1ss,Hsc2ss] # prepend 0 to make it 1-indexed
    carb_doses = [(0, 2e4), (800, 2e4)]
    uG = 0.0
    uI = 0
    ode_func = lambda t, x: model(t, x, 0, BW, Gb, carb_doses, uI, uG, 0)
    r = ode(ode_func)
    r.set_initial_value(x0)

        
    result = []
    while r.t < 500:
        result.append(r.integrate(r.t + 1))
    result = np.array(result)
    Gp = result[:, 1]
    t_vals = np.arange(0, 500, 1)
    plt.plot(t_vals, Gp)
    plt.show()
    breakpoint()