import numpy as np


class BodyModel:

    BW = 78
    Gb = 130

    # Glucose subsystem
    Vg = 1.49
    k1 = 0.065
    k2 = 0.079

    # Insuline subsystem
    VI = 0.04
    m1 = 0.379
    m2 = 0.673
    m3 = None
    m4 = 0.269
    m5 = 0.0526
    m6 = 0.8118
    HEb = 0.112

    # Endogenous glucose production
    kp1 = 2.7
    kp2 = 0.0021
    kp3 = 0.009
    kp4 = 0.0786
    ki = 0.0066

    # Glucose rate of appearance
    kmax = 0.0465
    kmin = 0.0076
    kabs = 0.023
    kgri = 0.0465
    f = 0.9
    a = 0.00016
    b = 0.68
    c = 0.00023
    d = 0.009

    # Glucose utilization
    Fcns = 1
    Vm0 = 4.65
    Vmx = 0.034
    Km0 = 471.13
    P2u = 0.084

    # Insuline secretion
    K = 0.99
    Alpha = 0.013
    Beta = 0.05
    Gamma = 0.5

    # Renal excretion
    ke1 = 0.0007
    ke2 = 269

    # Subcatenuous Insuln
    kd = 0.0164
    ka1 = 0.0018
    ka2 = 0.0182

    # Glucagon system
    Delta = 0.682
    Sigma = 1.093
    n = 0.15
    Zeta = 0.009
    Rho = 0.57
    kH = 0.16
    Hb = 93

    # Subcatenuous Glucagon
    kh1 = 0.0164
    kh2 = 0.0018
    kh3 = 0.0182

    # Basal States
    Gpb = None
    Gtb = None
    Ilb = None
    Ipb = None
    Ipob = None
    Ib = None
    IIRb = None
    Isc1ss = None
    Isc2ss = None
    kp1 = None
    Km0 = None
    SRHb = None
    Gth = None
    SRsHb = None
    XHb = None
    Ith = None
    IGRb = None
    Hsc1ss = None
    Hsc2ss = None

    def __init__(self, BW, Gb):
        self.BW = BW
        self.Gb = Gb
        self.set_basal_states()

    def maxfunc(self, x, a, ep):
        return (x + np.sqrt(x**2 + ep)) / 2

    def hill(self, t, tau, A, k):
        return A / (1 + np.exp(-k * (t - tau)))

    def delta(self, t, tau, A, k):
        Hval = A / (1 + np.exp(-k * (t - tau)))
        return k * Hval * (1 - Hval / A)

    def set_basal_states(self):
        Sb = 0
        self.IIRb = 0
        self.m3 = self.HEb * self.m1 / (1 - self.HEb)
        self.Ipb = self.IIRb / (self.m2 + self.m4 - (self.m1 * self.m2) / (self.m1 + self.m3))
        self.Ilb = self.Ipb * (self.m2 / (self.m1 + self.m3))
        self.Ib = self.Ipb / self.VI
        self.Ipob = 0
        self.EGPb = 2.4
        self.Gpb = self.Gb * self.Vg
        self.Gtb = (self.Fcns - self.EGPb + self.k1 * self.Gpb) / self.k2
        self.Isc1ss = self.IIRb / (self.kd + self.ka1)
        self.Isc2ss = self.kd * self.Isc1ss / self.ka2
        self.kp1 = self.EGPb + self.kp2 * self.Gpb + self.kp3 * self.Ib
        self.Km0 = (self.Vm0 * self.Gtb) / (self.EGPb - self.Fcns) - self.Gtb
        self.SRHb = self.n * self.Hb
        self.Gth = self.Gb
        self.SRsHb = max(self.Sigma * (self.Gth - self.Gb) + self.SRHb, 0)
        self.XHb = 0
        self.Ith = self.Ib
        self.IGRb = 0
        self.Hsc1ss = self.IGRb / (self.kh1 + self.kh2)
        self.Hsc2ss = self.kh1 * self.Hsc1ss / self.kh3

    def model(self, t, x, carbs, uI=0.0, uG=0.0):

        if carbs:
            D = carbs * 1000  # g to mg
        else:
            D = 1e-9
        # TODO should we manually deduct from t?

        # Load previous vector values
        G = x[0]
        Gp = x[1]
        Gt = x[2]
        Il = x[3]
        Ip = x[4]
        I = Ip / self.VI
        I1 = x[5]
        Id = x[6]
        Qsto1 = x[7]
        Qsto2 = x[8]
        Qgut = x[9]
        Ra = self.f * self.kabs * Qgut / self.BW
        X = x[10]
        Uii = self.Fcns
        Uid = (self.Vm0 + self.Vmx * X) * Gt / (self.Km0 + Gt)
        SRsH = x[11]
        H = x[12]
        XH = x[13]
        EGP = self.kp1 - self.kp2 * Gp - self.kp3 * Id + self.Zeta * XH
        E = self.ke1 * (Gp - self.ke2) * self.hill(Gp, self.ke2, 1, 4)
        Isc1 = x[14]
        Isc2 = x[15]
        Rai = self.ka1 * Isc1 + self.ka2 * Isc2
        Hsc1 = x[16]
        Hsc2 = x[17]
        Rah = self.kh3 * Hsc2

        # Set new vector values
        xdot = np.zeros(len(x))
        xdot[1] = EGP + Ra - Uii - E - self.k1 * Gp + self.k2 * Gt
        xdot[2] = -Uid + self.k1 * Gp - self.k2 * Gt
        Gdot = xdot[1] / self.Vg
        xdot[0] = Gdot
        HE = self.HEb
        self.m3 = HE * self.m1 / (1 - HE)
        xdot[3] = -(self.m1 + self.m3) * Il + self.m2 * Ip
        xdot[4] = -(self.m2 + self.m4) * Ip + self.m1 * Il + Rai
        xdot[5] = -self.ki * (I1 - I)
        xdot[6] = -self.ki * (Id - I)
        Qsto = Qsto1 + Qsto2
        kempt = self.kmin + (self.kmax - self.kmin) / 2 * (
            np.tanh(5 / (2 * D * (1 - self.b)) * (Qsto - self.b * D))
            - np.tanh(5 / (2 * D * self.c) * (Qsto - self.c * D))
            + 2
        )
        xdot[7] = -self.kgri * Qsto1 + D * self.delta(t, 60, 1, 4)
        xdot[8] = -kempt * Qsto2 + self.kgri * Qsto1
        xdot[9] = -self.kabs * Qgut + kempt * Qsto2
        xdot[10] = -self.P2u * (X - I + self.Ib)
        SRdH = self.Delta * self.maxfunc(-Gdot, 0, 0.0001)
        SRH = SRsH + SRdH
        xdot[11] = -self.Rho * (
            SRsH
            - self.maxfunc(
                (self.Sigma * (self.Gth - G) / (self.maxfunc(I - self.Ith, 0, 0.0001) + 1))
                + self.SRHb,
                0,
                0.0001,
            )
        )
        xdot[12] = -self.n * H + SRH + Rah
        xdot[13] = -self.kH * XH + self.kH * self.maxfunc(H - self.Hb, 0, 0.0001)
        xdot[14] = (
            -(self.kd + self.ka1) * Isc1
            + self.IIRb
            + (1 / 78) * uI * 6944.4 * self.delta(t, 30, 1, 4)
        )
        xdot[15] = self.kd * Isc1 - self.ka2 * Isc2
        xdot[16] = -(self.kh1 + self.kh2) * Hsc1 + (1 / 78) * uG * 1e6 * self.delta(t, 150, 1, 4)
        xdot[17] = self.kh1 * Hsc1 - self.kh3 * Hsc2

        return xdot
