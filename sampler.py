from scipy.io import wavfile
class Drums:
    def __init__(self, kick: str, snare: str, hihat: str) -> None:
        self.kick = wavfile.read('./samples/kick.WAV')
        self.snare = wavfile.read('./samples/snare.WAV')
        self.hihat = wavfile.read('./samples/hihat.WAV')




arr = wavfile.read('./samples/kick.WAV')
arr.