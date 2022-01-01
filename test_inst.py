import json
import numpy as np
from scipy.signal import signaltools
from configs import SAMPLE_RATE
import instruments as inst
import effects as eff
import cProfile
import matplotlib.pyplot as plt
from scipy.io import wavfile


env = inst.Envelope([0.05, 0.08, 0.50, .05], mode='adsr')

osc1 = inst.Oscillator('m_saw', 0.7)
osc2 = inst.Oscillator('triangle', None, 1)
# osc3 = inst.Oscillator('m_square', 0.75)

harm = inst.SynthBase('lead', env, [osc1, osc2], [2, 1], volume = .6)
harm.plot()

filter = eff.BandPass(lowcut = 1000, highcut = 4000, strength = 3)

harm = inst.Instrument(harm, [filter])

harm.save_json('./instruments/lead.json')
arr = harm.play_note(1, 45+12)
# arr += harm.play_note(1, 49+12)
# arr += harm.play_note(1, 52+12)
# arr += harm.play_note(1, 56+12)

wavfile.write('./out/harm.wav', SAMPLE_RATE, arr)
