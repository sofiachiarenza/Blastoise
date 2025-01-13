import sys
from fftlog import *
import numpy as np
import time
from calculator_fkem import CalculatorFKEM
import os


# First, generate benchmarks if you haven't done so yet.
if not os.path.isfile("outputs/fkem_clgg.npz"):
    cal_nl = CalculatorFKEM("./config_nonlim_fang.yml")
    cal_nl.setup()
    cal_nl.run()
    cal_nl.write_output()
    cal_nl.teardown()
