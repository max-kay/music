import numpy as np
from numpy.random.mtrand import sample
from scipy.io import wavfile
import matplotlib.pyplot as plt
from configs import SAMPLE_RATE
import song

sm, arr = wavfile.read('./out/test.wav')

loops = 5
tot_len = 8*SAMPLE_RATE*(loops-1)+len(arr)

out = np.zeros(8*SAMPLE_RATE*(loops-1)+len(arr))

for i in range(loops):
    current = np.pad(arr, (i*8*SAMPLE_RATE, tot_len - i*8*SAMPLE_RATE- len(arr)))
    out = current + out

# fig, ax = plt.subplots()
# plt.plot(arr)
# plt.show()

wavfile.write('./out/out.wav', sm, out)