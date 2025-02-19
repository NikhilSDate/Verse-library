import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import random

# implementation from https://github.com/jonasnm/svelte-flask-hovorka-simulator/blob/master/hovorka_simulator.py

# matlab code refers to parameters from here https://pmc.ncbi.nlm.nih.gov/articles/PMC2825634/

class HovorkaModel:
    
    num_variables = 11
    
    '''
    BW is body weight in kilograms
    Gb is basal glucose in mg/dL
    '''
    
    eInsSub1 = 1
    eInsSub2 = 2
    eInsPlas = 3
    eInsActT = 4
    eInsActD = 5
    eInsActE = 6
    eGutAbs = 7
    eGluPlas = 8
    eGluComp = 9
    eGluInte = 10
    eGluMeas = 11

    def __init__(self, param, meal_plan=None, exercise_plan=None, options=None):
        self.meal_plan = meal_plan
        self.exercise_plan = exercise_plan

        # Default options
        self.opt = {
            "name": "HovorkaPatient",
            "patient": ["patientAvg"],
            "sensorNoiseType": "none",
            "sensorNoiseValue": 0.4,
            "intraVariability": 0.0,
            "mealVariability": 0.0,
            "basalGlucose": 6.5,
            "randomInitialConditions": False,
            "initialGlucose": np.nan,
            "initialInsulinOnBoard": np.nan,
            "initialState": [],
            "useTreatments": True,
            "treatmentRules": [{
                "sensorGlucose": 3.9,
                "bloodGlucose": 2.8,
                "duration": 15,
                "lastTreatment": 40
            }],
            "wrongPumpParam": False,
            "pumpBasalsError": {"time": [], "value": []},
            "carbFactorsError": {"time": [], "value": []},
            "carbsCountingError": False,
            "carbsCountingErrorValue": {"bias": [], "std": []},
            "dailyCarbsCountingError": {"time": [], "value": []},
            "RNGSeed": -1
        }

        if options:
            self.opt.update({k: options[k] for k in self.opt if k in options})

        self.name = self.opt["name"]
        random.seed(self.opt["RNGSeed"]) if self.opt["RNGSeed"] > 0 else None

        self.param = param

        if self.opt["basalGlucose"] < 0:
            self.param["GBasal"] = np.random.normal(6.5, 1.0)
            while self.param["GBasal"] < 5 or self.param["GBasal"] > 8:
                self.param["GBasal"] = np.random.normal(6.5, 1.0)
        else:
            self.param["GBasal"] = self.opt["basalGlucose"]


        if "RCl" not in self.param:
            self.param["RTh"] = 14
            self.param["RCl"] = np.random.lognormal(np.log(1 / (2 * 60)), 0.2)

        if "TGlu" not in self.param:
            self.param["TauGlu"] = np.random.lognormal(np.log(19), 0.2)
            self.param["TGlu"] = np.random.lognormal(np.log(0.0012), 0.2)
            self.param["MCRGlu"] = np.random.lognormal(np.log(0.012), 0.2)

        if self.opt["randomInitialConditions"]:
            self.opt["initialGlucose"] = self.param["GBasal"] * (1 + 2.0 * np.random.randn())
            while self.opt["initialGlucose"] < 4 or self.opt["initialGlucose"] > 12:
                self.opt["initialGlucose"] = self.param["GBasal"] * (1 + 2.0 * np.random.randn())

            self.opt["initialInsulinOnBoard"] = 0.1 * self.param["TDD"] * (np.random.rand() - 0.5)
            while self.opt["initialInsulinOnBoard"] < -0.8 * self.param["Ub"]:
                self.opt["initialInsulinOnBoard"] = 0.1 * self.param["TDD"] * (np.random.rand() - 0.5)
        else:
            self.opt["initialGlucose"] = self.param["GBasal"] if np.isnan(self.opt["initialGlucose"]) else self.opt["initialGlucose"]
            self.opt["initialInsulinOnBoard"] = 0.0 if np.isnan(self.opt["initialInsulinOnBoard"]) else self.opt["initialInsulinOnBoard"]

        self.param["carbFactors"] = {"value": self.param["carbF"], "time": 0}
        self.param["pumpBasals"] = {"value": self.param["Ub"], "time": 0}

        self.apply_wrong_pump_param()
        self.apply_carbs_counting_errors()

        self.CGM = {
            "lambda": 15.96,
            "epsilon": -5.471,
            "delta": 1.6898,
            "gamma": -0.5444,
            "error": 0
        }

        self.meals = []
        self.glucagon = []

        self.variability = {key: {"val": self.param[key], "target": self.param[key]} for key in [
            "EGP0", "F01", "k12", "ka1", "ka2", "ka3", "St", "Sd", "Se", "ka", "ke"
        ]}
        
        for k, v in self.variability.items():
            self.variability[k]['val'] = self.param[k]
            self.variability[k]['target'] = self.param[k]

        self.stateScale = np.array([
            1.0, 1.0, 1e-6 * 60 * self.param["ke"] * (self.param["Vi"] * self.param["w"]),
            1.0, 1.0, 1.0, 60 / self.param["Vg"], 1 / self.param["Vg"],
            1 / self.param["Vg"], 1.0, 1.0
        ])

    def load_patient_parameters(self, patient_name):
        pass

    def apply_wrong_pump_param(self):
        pass

    def apply_carbs_counting_errors(self):
        pass

    def apply_intra_variability(self, t):
        for key in self.variability:
            if isinstance(self.variability[key], dict):
                self.variability[key]["target"] = self.param[key]

        if self.opt["intraVariability"] > 0:
            for key in self.variability:
                if isinstance(self.variability[key], dict):
                    self.variability[key]["target"] = self.param[key] * (
                        1 + 0.2 * self.opt["intraVariability"] * np.sin(2 * np.pi * (t + self.variability[key]["phase"]) / self.variability[key]["period"])
                    )

        exerc_int = 0
        exerc_type = "aerobic"
        for exercise in self.exercises:
            if exercise["time"] <= t < exercise["time"] + exercise["duration"]:
                exerc_int = exercise["intensity"]
                exerc_type = ExercisePlan.types_of_exercise[exercise["type"]]

        if exerc_int > 0:
            self.variability["ka"]["target"] = self.param["ka"] * (1 + 2 * exerc_int)
            self.variability["ka1"]["target"] = self.param["ka1"] * (1 + 4 * exerc_int)
            self.variability["ka2"]["target"] = self.param["ka2"] * (1 + 4 * exerc_int)
            self.variability["ka3"]["target"] = self.param["ka3"] * (1 + 4 * exerc_int)

            if exerc_type == "mixed":
                mixing_effect = -0.7 + (0.7 + 0.7) * np.random.rand()
                mixing_coeff = [1 + mixing_effect, 1 - mixing_effect]
            else:
                mixing_coeff = [1, 1]

            if exerc_type in ["aerobic", "mixed"]:
                self.variability["St"]["target"] = self.param["St"] * (1 + 5 * mixing_coeff[0] * exerc_int)
                self.variability["Sd"]["target"] = self.param["Sd"] * (1 + 10 * mixing_coeff[0] * exerc_int)

            if exerc_type in ["anaerobic", "mixed"]:
                self.variability["EGP0"]["target"] = self.param["EGP0"] * (1 + mixing_coeff[1] * exerc_int)
                self.variability["Se"]["target"] = self.param["Se"] / (1 + 6 * mixing_coeff[1] * exerc_int)

        alpha = 0.7
        for key in self.variability:
            if isinstance(self.variability[key], dict):
                self.variability[key]["val"] = (1 - alpha) * self.variability[key]["val"] + alpha * self.variability[key]["target"]

    
    
    def gut2comp_model(self, t, meal):
        if t > (meal["time"] + meal["Delay"]):
            return (1e6 / (self.param["w"] * self.param["MCHO"])) * meal["Bio"] * meal["value"] * \
                (t - meal["time"] - meal["Delay"]) * np.exp(-(t - meal["time"] - meal["Delay"]) / meal["TauM"]) / meal["TauM"]**2
        return 0

    def _get_initial_state(self, opt):
        initial_state = np.zeros(self.eGluMeas + 1)
        Gs0 = opt["initialGlucose"]
        initial_state[self.eGluPlas] = Gs0 * self.param["Vg"]
        initial_state[self.eGluMeas] = Gs0
        initial_state[self.eGluInte] = Gs0

        Qb = opt["initialInsulinOnBoard"] if "initialInsulinOnBoard" in opt else 0

        initial_state[self.eInsPlas] = (self.param["Ub"] + Qb) / (
            self.param["ke"] / (1e6 * self.param["ka"] / (self.param["Vi"] * self.param["w"])))

        initial_state[self.eInsActT] = self.param["St"] * initial_state[self.eInsPlas]
        initial_state[self.eInsActD] = self.param["Sd"] * initial_state[self.eInsPlas]
        initial_state[self.eInsActE] = self.param["Se"] * initial_state[self.eInsPlas]

        Q10 = Gs0 * self.param["Vg"]
        Q20 = Q10 * initial_state[self.eInsActT] / (initial_state[self.eInsActD] + self.param["k12"])
        initial_state[self.eGluComp] = Q20

        initial_state[self.eInsSub1] = self.param["Ub"] / 60 / self.param["ka"]
        initial_state[self.eInsSub2] = initial_state[self.eInsSub1]

        initial_state[self.eGutAbs] = 0

        return initial_state

    def model(self, t, y, u):
        dydt = np.zeros_like(y)

        dydt[self.eInsSub1] = u - y[self.eInsSub1] * self.variability["ka"]["val"] # changed u / 60 to u since we are sending raw insulin amounts
        dydt[self.eInsSub2] = y[self.eInsSub1] * self.variability["ka"]["val"] - y[self.eInsSub2] * self.variability["ka"]["val"]

        GluPlas = sum(
            (1e6 / (self.param["w"] * self.param["MCRGlu"])) * g["value"] *
            (t - g["time"]) * np.exp(-(t - g["time"]) / self.param["TauGlu"]) / self.param["TauGlu"]**2
            for g in self.glucagon
        )

        dydt[self.eInsPlas] = 1e6 * y[self.eInsSub2] * self.variability["ka"]["val"] / (self.param["Vi"] * self.param["w"]) - \
                            y[self.eInsPlas] * self.variability["ke"]["val"]

        dydt[self.eInsActT] = -self.variability["ka1"]["val"] * y[self.eInsActT] + \
                            self.variability["ka1"]["val"] * self.variability["St"]["val"] * y[self.eInsPlas]
        dydt[self.eInsActD] = -self.variability["ka2"]["val"] * y[self.eInsActD] + \
                            self.variability["ka2"]["val"] * self.variability["Sd"]["val"] * y[self.eInsPlas]
        dydt[self.eInsActE] = -self.variability["ka3"]["val"] * y[self.eInsActE] + \
                            self.variability["ka3"]["val"] * self.variability["Se"]["val"] * y[self.eInsPlas]

        Um = sum(self.gut2comp_model(t, meal) for meal in self.meals)


        GluPlas_exp = np.exp(-1 / (GluPlas * self.param["TGlu"])) if GluPlas != 0 else 0
        dydt[self.eGluPlas] = -((self.variability["F01"]["val"] / 0.85) / (y[self.eGluPlas] + self.param["Vg"]) + y[self.eInsActT]) * y[self.eGluPlas] + \
                            self.variability["k12"]["val"] * y[self.eGluComp] - \
                            self.param["RCl"] * (y[self.eGluPlas] - self.param["RTh"] * self.param["Vg"]) * \
                            (y[self.eGluPlas] > self.param["RTh"] * self.param["Vg"]) + \
                            self.variability["EGP0"]["val"] * (np.exp(-y[self.eInsActE]) + GluPlas_exp) + Um

        dydt[self.eGluComp] = y[self.eInsActT] * y[self.eGluPlas] - \
                            (self.variability["k12"]["val"] + y[self.eInsActD]) * y[self.eGluComp]

        dydt[self.eGluInte] = (y[self.eGluPlas] / self.param["Vg"] - y[self.eGluInte]) / self.param["TauS"]

        return dydt
    
    def construct_meal(self, meal):
        return {
            'time': meal.time,
            'value': meal.carbs,
            'Delay': 0,
            'TauM': self.param['TauM'],
            'Bio': self.param['Bio']
        }
        
    def set_meals(self, meals):
        self.meals = [self.construct_meal(meal) for meal in meals] 
    
    def get_init_state(self, G):
        G = self.mgdl_to_mmol(G)
        return self._get_initial_state({'initialGlucose': G})
    
    def get_init_range(self, Gl, Gh):
        return [self.get_init_state(Gl), self.get_init_state(Gh)]
    
    def mmol_to_mgdl(self, G):
        return G * 18
    
    def mgdl_to_mmol(self, G):
        return G / 18
    
def patient_original(opt):
    param = {
        "MCHO": 180.1577,
        "w": 74.9,
        "TauS": np.exp(2.372),
        "EGP0": 16.9,
        "F01": 11.1,
        "k12": 0.060,
        "RTh": 9,
        "RCl": 0.01,
        "ka1": 0.0034,
        "ka2": 0.056,
        "ka3": 0.024,
        "St": 18.41e-4,
        "Sd": 5.05e-4,
        "Se": 190e-4,
        "ka": 0.018,
        "ke": 0.14,
        "Vi": 120,
        "Vg": 160,
        "Bio": 0.8,
        "TauM": 1 / 0.025,
        "TauGlu": 19,
        "TGlu": 0.0012,
        "MCRGlu": 0.012
    }
    Gs0 = opt["basalGlucose"]
    Q10 = Gs0 * param["Vg"]
    Fn = Q10 * (param["F01"] / 0.85) / (Q10 + param["Vg"])
    Fr = param["RCl"] * (Q10 - param["RTh"] * param["Vg"]) * (Q10 > param["RTh"] * param["Vg"])
    
    coefficients = [
        -Q10 * param["St"] * param["Sd"] - param["EGP0"] * param["Sd"] * param["Se"],
        -param["k12"] * param["EGP0"] * param["Se"] + (param["EGP0"] - Fr - Fn) * param["Sd"],
        param["k12"] * (param["EGP0"] - Fn - Fr)
    ]
    roots_solution = np.roots(coefficients)
    real_roots = [r.real for r in roots_solution if np.isreal(r)]
    
    if not real_roots:
        raise ValueError()
    
    initial_guess = max(real_roots)
    
    def insulin_equation(x):
        return (-Fn - Q10 * param["St"] * x + param["k12"] * (Q10 * param["St"] * x) / (param["k12"] + param["Sd"] * x) - Fr + param["EGP0"] * np.exp(-param["Se"] * x))
    
    Ip0 = fsolve(insulin_equation, initial_guess)[0]
    
    param["Ub"] = 60 * Ip0 * param["ke"] / (1e6 / (param["Vi"] * param["w"]))
    param["carbF"] = min(max(round(2 * (param["MCHO"] * (0.4 * max(param["St"], 16e-4) + 0.6 * min(max(param["Sd"], 3e-4), 12e-4)) * Gs0 * param["Vg"])/(param["ke"] * param["Vi"]))/2, 2), 25)
    param["TDD"] = min(max(round(param["Ub"] * 24 + 200 / param["carbF"], 2), 10), 110)
    return param
    
        
    
    
        
    