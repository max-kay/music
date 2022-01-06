import numpy as np
from scipy import signal
from numpy.random import rand

sine = lambda arr, mod: np.sin(arr)
square = lambda arr, mod: signal.square(arr)
saw =  lambda arr, mod : signal.sawtooth(arr)
triangle = lambda arr, mod: signal.sawtooth(arr, 0.5)
noise = lambda arr, mod: rand(len(arr)) * 2 - 1
m_saw = lambda arr, mod: signal.sawtooth(arr, mod)
m_square = lambda arr, mod: signal.square(arr, mod)

def get_stammfunc(mode: str):
    if mode == 'sine':
        return lambda arr, mod: -np.cos(arr)
    if mode == 'square':
        return triangle
    if mode == 'saw':
        return lambda arr, mod: np.power(triangle(arr/2, mod), 2)*2 - 1
    if mode == 'triangle':
        return lambda arr, mod: np.power(m_saw(arr, 0.5)+ 1, 2)/2 - 1
    if mode == 'noise':
        return noise
    if mode == 'm_saw':
        return lambda arr, mod: np.power(m_saw(arr, mod)+ 1, 2)/2 - 1
    if mode == 'm_square':
        return m_saw
