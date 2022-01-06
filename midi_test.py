import mido
import numpy as np
from scipy.io import wavfile
from music.configs import SAMPLE_RATE
from music import player


FILE_PATH = './midi_files/seven8.mid'

alles = player.Player.from_midi(FILE_PATH)

alles.save_json()

alles.save_wav()
