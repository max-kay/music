import numpy as np

from scipy import signal
import scipy
from scipy.signal import signaltools

from . import array_func as af
from . import oscillators as osc

from .configs import SAMPLE_RATE

medfilt = lambda arr, size: signaltools.medfilt(arr, kernel_size=size*2+1)

def band_pass(arr: np.ndarray, lowcut, highcut, strength) -> np.ndarray:
    N, Wn = signal.buttord([lowcut, highcut], [lowcut*2**-0.2, highcut*2**0.2], 3, strength+3, fs = SAMPLE_RATE)
    sos = signal.butter(N, Wn, 'band', output='sos', fs=SAMPLE_RATE)
    arr = signal.sosfilt(sos, arr)
    return arr

def low_pass(arr: np.ndarray, cutoff, slope, strength) -> np.ndarray:#TODO
    N, Wn = signal.buttord([10, cutoff], [1, cutoff*2**(slope/10)], 3, strength+3, fs = SAMPLE_RATE)
    sos = signal.butter(N, Wn, 'band', output='sos', fs=SAMPLE_RATE)
    arr = signal.sosfilt(sos, arr)
    return arr

def high_pass(arr: np.ndarray, cutoff, slope, strength) -> np.ndarray:#TODO
    N, Wn = signal.buttord([cutoff, 30_000], [cutoff*2**(slope/10), 40_000], 3, strength+3, fs = SAMPLE_RATE)
    sos = signal.butter(N, Wn, 'band', output='sos', fs=SAMPLE_RATE)
    arr = signal.sosfilt(sos, arr)
    return arr

def comb_delay(arr: np.ndarray, delta_ms: float, gain: float, loops = 5) -> np.ndarray:
    out = np.zeros(len(arr))
    for i in range(loops):
        out = af.add_with_index(out, arr*(gain**i), round(delta_ms*SAMPLE_RATE/1000)*(i+1))
    return out

def all_pass_delay(arr: np.ndarray, delta_ms: float, gain: float, loops = 20) -> np.ndarray:
    out = np.zeros(SAMPLE_RATE)
    for i in range(loops):
        out = af.add_with_index(out, arr*(gain**i), round(delta_ms*SAMPLE_RATE/1000)*(i+1))
    return af.add_dif_len(out*(1-gain**2), -gain*arr)

def reverb(arr: np.ndarray, wet=1, dry=1) -> np.ndarray:
    a = .7
    parallel = [(a+.042, 4.799), (a+.033, 4.999), (a+.015, 5.399), (a-0.003, 5.801)]
    series = [(a, 1.051), (a, 0.337)]
    out = np.zeros(SAMPLE_RATE)
    for gain, delta in parallel:
        out = af.add_dif_len(out, comb_delay(arr, gain, delta))
    out = out/5000
    for gain, delta in series:
        out = all_pass_delay(out, gain, delta)
    return af.add_dif_len(out * wet, arr * dry)/3

def luis(arr: np.ndarray) -> np.ndarray:
    time = np.linspace(0, 120, 400)
    wave = osc.sine(time, 0)
    arr = af.add_dif_len(signal.convolve(wave, arr), arr)
    return arr

def dist(arr: np.ndarray) -> np.ndarray:
    return osc.triangle(arr, 0)
