import numpy as np
import music.instruments as inst
import music.array_func as af
from music import effects

instrument = inst.Synthesizer.from_json('instruments/clean_test.json')

bass = 60

arr = instrument.play_note(5, bass)
# arr += harm.play_note(5, bass + 7)
# arr += harm.play_note(5, bass + 7 + 9)
# arr += harm.play_note(5, bass + 7 + 9 + 7)
af.save_wav('out/clean_test.wav', arr)
