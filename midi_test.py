import mido
import numpy as np
from configs import SAMPLE_RATE
import song
from scipy.io import wavfile
import instruments as inst


filepath = './midi_files/vel_test.mid'


alles = song.Song.from_midi(filepath)


arr = alles.play()

arr = np.trim_zeros(arr, 'b')


wavfile.write(f'./out/test.wav', SAMPLE_RATE, arr)
