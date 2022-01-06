import numpy as np

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

def cut_or_pad(length: int, volatile: np.ndarray, pad_val=0):
    if len(volatile) > length:
        return volatile[:length]
    return np.pad(volatile, (0, length -len(volatile)), constant_values=(0, pad_val))
