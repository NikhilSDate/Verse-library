import matlab.engine
from dotenv import load_dotenv
import os
load_dotenv()

eng = matlab.engine.start_matlab()
eng.addpath(os.environ['MATLAB_PATH'])
eng.simulation()