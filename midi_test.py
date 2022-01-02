import mido
import numpy as np
from scipy.io import wavfile
from configs import SAMPLE_RATE
import song
import instruments as inst


FILE_PATH = './midi_files/test.mid'


alles = song.Song.from_midi(FILE_PATH,)


arr = alles.play()

arr = np.trim_zeros(arr, 'b')


wavfile.write('./out/testdgv.wav', SAMPLE_RATE, alles.play())
