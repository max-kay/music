import numpy as np
from instruments import load_wav

arr = load_wav('./samples/kick.wav', 1)
print(np.max(arr), np.min(arr))