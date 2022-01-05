import numpy as np
import scipy.signal as signal
from numpy.random import rand

sine = lambda arr, mod: np.sin(arr)
square = lambda arr, mod: signal.square(arr)
saw =  lambda arr, mod : signal.sawtooth(arr)
triangle = lambda arr, mod: signal.sawtooth(arr, 0.5)
noise = lambda arr, mod: rand(len(arr)) * 2 - 1
m_saw = lambda arr, mod: signal.sawtooth(arr, mod)
m_square = lambda arr, mod: signal.square(arr, mod)
