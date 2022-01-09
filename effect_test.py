from music import effects as eff
from music.array_func import load_wav, save_wav

sound = load_wav('./samples/sound.wav')

sound = eff.dist(sound)
save_wav('./out/sound.wav', sound)
