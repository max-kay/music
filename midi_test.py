import mido
import numpy as np
from scipy.io import wavfile
from configs import SAMPLE_RATE
import player
import instruments as inst


FILE_PATH = './midi_files/test.mid'

alles = player.Player.from_midi(FILE_PATH)

alles.save_json()

alles.save_wav()
