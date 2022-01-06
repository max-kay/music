import cProfile
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
from music.configs import SAMPLE_RATE
import music.instruments as inst


instrument = inst.Synthesizer.from_json('instruments/test_instr2.json')

bass = 50

arr = instrument.play_note(5, bass)
# arr += harm.play_note(5, bass + 7)
# arr += harm.play_note(5, bass + 7 + 9)
# arr += harm.play_note(5, bass + 7 + 9 + 7)

wavfile.write(f'./out/{instrument.name}.wav', SAMPLE_RATE, arr)


# instrument.plot()
# plt.show()
