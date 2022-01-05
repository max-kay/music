import numpy as np

from scipy import signal
from scipy.signal import signaltools

import array_func as af

from configs import SAMPLE_RATE

medfilt = lambda arr, size: signaltools.medfilt(arr, kernel_size=size*2+1)

def band_pass(arr: np.ndarray, lowcut, highcut, strength) -> np.ndarray:
    N, Wn = signal.buttord([lowcut, highcut], [lowcut*2**-0.3, highcut*2**0.3], 3, strength, fs = SAMPLE_RATE*2)
    sos = signal.butter(N, Wn, 'band', output='sos', fs=SAMPLE_RATE*2)
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
    out = out/50000000
    for gain, delta in series:
        out = all_pass_delay(out, gain, delta)
    return af.add_dif_len(out * wet, arr * dry)/3
