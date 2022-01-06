import matplotlib.pyplot as plt
from music.configs import SAMPLE_RATE
from music.oscillators import m_saw
from music.instruments import get_default_time_arr

arr = m_saw(10*get_default_time_arr(2), .3)

new = []
sum = 0
for element in arr:
    sum += element/SAMPLE_RATE
    new.append(sum)

fig, ax = plt.subplots()
plt.plot(arr)
plt.plot(new)
plt.show()
