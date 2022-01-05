from matplotlib import pyplot as plt
import numpy as np
from scipy.io import wavfile
import cProfile
from configs import SAMPLE_RATE
import instruments as inst


harm = inst.Synthesizer.from_json('instruments/test_inst.json')

bass = 57

arr = harm.play_note(5, bass)
# arr += harm.play_note(5, bass + 7)
# arr += harm.play_note(5, bass + 7 + 9)
# arr += harm.play_note(5, bass + 7 + 9 + 7)

wavfile.write(f'./out/{[eff.__str__() for eff in harm.effects]}.wav', SAMPLE_RATE, arr)

harm.plot()
plt.show()
