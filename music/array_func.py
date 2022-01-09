import numpy as np
from scipy.io import wavfile

from .configs import SAMPLE_RATE

def add_with_index(fix: np.ndarray, mobile: np.ndarray, index) -> np.ndarray:
    if len(mobile) + index > len(fix):
        fix = np.pad(fix, (0, len(mobile) + index - len(fix)), constant_values=(0, 0))
    for i, val in zip(range(index, index+len(mobile)), mobile):
        fix[i] += val
    return fix

def add_dif_len(arr1, arr2):
    if len(arr1) < len(arr2):
        arr1 = np.pad(arr1, (0, len(arr2)-len(arr1)), constant_values=(0, 0))
    else:
        arr2 = np.pad(arr2, (0, len(arr1)-len(arr2)), constant_values=(0, 0))
    return np.add(arr1, arr2)

def cut_or_pad(length: int, volatile: np.ndarray, mode = 'val', pad_val = 0):
    if len(volatile) > length:
        return volatile[:length]
    if mode == 'last':
        return np.pad(volatile, (0, length -len(volatile)), constant_values=(0, volatile[-1]))
    if mode == 'zero':
        return np.pad(volatile, (0, length -len(volatile)), constant_values=(0, 0))
    if mode == 'val':
        return np.pad(volatile, (0, length -len(volatile)), constant_values=(0, pad_val))



def load_wav(file_path: str):#TODO
    samplerate, arr = wavfile.read(file_path)
    assert samplerate == SAMPLE_RATE
    max_val = np.iinfo(arr.dtype).max
    # arr = arr[:,0]/max_val
    return arr

def save_wav(file_path: str, arr: np.ndarray, dtype = 'int32') -> None:
    dtype = np.dtype(dtype)
    data = arr * np.iinfo(dtype).max *0.1
    data = data.astype(dtype)
    wavfile.write(file_path, SAMPLE_RATE, data)

def get_default_time_arr(duration):
    return np.linspace(0, duration, round(SAMPLE_RATE*duration))
