import numpy as np
from numpy.lib.function_base import extract
from scipy import signal
from numpy.random import rand

sine = lambda arr, mod: np.sin(arr)
square = lambda arr, mod: signal.square(arr)
saw =  lambda arr, mod : signal.sawtooth(arr)
triangle = lambda arr, mod: signal.sawtooth(arr, 0.5)
noise = lambda arr, mod: rand(len(arr)) * 2 - 1
m_saw = lambda arr, mod: signal.sawtooth(arr, mod)
m_square = lambda arr, mod: signal.square(arr, mod)


def quadratic_base(arr, mod) -> np.ndarray: #taken from scipy wavefroms and modified
    t, w = np.asarray(arr), np.asarray(mod)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    # tsub * (tsub - 2*np.pi * wsub)
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = extract(mask2, tmod)
    wsub = extract(mask2, w)
    ysub = tsub * (tsub -  2*np.pi * wsub)/(np.pi*wsub)**2
    np.place(y, mask2, ysub)

    # on the interval width*2*pi to 2*pi function is
    # -(tsub**2 - 2*np.pi*(1 + wsub)*tsub + 4*np.pi**2 * wsub)

    mask3 = (1 - mask1) & (1 - mask2)
    tsub = extract(mask3, tmod)
    wsub = extract(mask3, w)
    ysub = (tsub**2 - 2*np.pi*(1 + wsub)*tsub + 4*np.pi**2 * wsub)/(np.pi**2*(4*wsub - (1 + wsub)**2))
    np.place(y, mask3, ysub)
    return y

quadratic = lambda arr, mod: quadratic_base(arr, 0.5)
m_quadratic = lambda arr, mod: quadratic_base(arr, mod)
quadratic_blipp = lambda arr, mod: quadratic_base(arr, 1)

def get_stamm_like(mode: str):
    if mode == 'sine':
        return sine
    if mode == 'square':
        return triangle
    if mode == 'saw':
        return quadratic_blipp
    if mode == 'triangle':
        return quadratic
    if mode == 'noise':
        return noise
    if mode == 'm_saw':
        return m_quadratic
    if mode == 'm_square':
        return m_saw
