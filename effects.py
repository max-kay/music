import numpy as np

from scipy import signal
from scipy.signal import signaltools

from configs import SAMPLE_RATE

medfilt = lambda arr, size: signaltools.medfilt(arr, kernel_size=size*2+1)

def band_pass(arr: np.ndarray, lowcut, highcut, strength) -> np.ndarray:
            N, Wn = signal.buttord([lowcut, highcut], [lowcut*2**-0.3, highcut*2**0.3], 3, strength, fs = SAMPLE_RATE*2)
            sos = signal.butter(N, Wn, 'band', output='sos', fs=SAMPLE_RATE*2)
            arr = signal.sosfilt(sos, arr)
            return arr


    