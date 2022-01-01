import json
import warnings
import matplotlib
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import fft, fftfreq

import effects as eff

from scipy.io import wavfile

from scipy import signal

from configs import CYCLES_FOR_PLOTS, FREQ_FOR_PLOTS, SAMPLE_RATE, ENV_LENGTH_FOR_PLOTS, CYCLES_FOR_FFT, TUNING

def load_wav(file_path: str, max_val):
    _, arr = wavfile.read(file_path)
    arr = arr[:,0]/max_val
    return arr

class Oscillator:
    modes = ['sine', 'square', 'saw', 'triangle', 'm_saw', 'm_square', 'noise']
    def __init__(self, mode: str, modulation = None, detune_oct = 0, detune_cent = 0) -> None:
        assert mode in self.modes, f'{mode} is undefined, try one of the following modes: \n {self.modes}'

        self.mode = mode
        self.modulation = modulation

        self.cents = detune_cent
        self.octave = detune_oct

        if mode == 'sine':
            self.func = lambda arr: np.sin(arr)
        if mode == 'square':
            self.func = lambda arr: signal.square(arr)
        if mode == 'saw':
            self.func = lambda arr: signal.sawtooth(arr)
        if mode == 'triangle':
            self.func = lambda arr: signal.sawtooth(arr, 0.5)
        if mode == 'noise':
            self.func = lambda arr: rand(len(arr)) * 2 - 1
        if mode == 'm_saw':
            self.func = lambda arr: signal.sawtooth(arr, modulation)
        if mode == 'm_square':
            self.func = lambda arr: signal.square(arr, modulation)
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        dict = {
            'mode': self.mode,
            'modulation': self.modulation,
            'detune_oct': self.octave,
            'detune_cent': self.cents
        }
        return dict
    
    def play_freq(self, duration, freq) -> np.ndarray:
        freq = (freq * 2**self.octave)  * 1.0005777895065548**self.cents
        arr = np.arange(round(duration * SAMPLE_RATE)) * 2 * np.pi / (SAMPLE_RATE/freq)
        wave = self.func(arr)
        return wave
    
    def plot(self, ax = None) -> None:
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.play_freq(CYCLES_FOR_PLOTS/FREQ_FOR_PLOTS, FREQ_FOR_PLOTS))
        ax.set_label(self.mode)

class Envelope:
    modes = ['adsr', 'ad']
    def __init__(self, envelope: tuple, mode='adsr') -> None:
        assert mode in self.modes, f'undefined mode, use {self.modes}'
        self.mode = mode
        self.envelope = envelope

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        dict = {
            'mode': self.mode,
            'envelope': self.envelope
            }
        return dict

    def get_envelope(self, duration = None) -> np.ndarray:
        if self.mode == 'adsr':
            val_attack, val_decay, val_sustain, val_release = self.envelope
            attack = np.linspace(0, 1, round(val_attack*SAMPLE_RATE))
            decay = np.linspace(1, val_sustain, round(val_decay*SAMPLE_RATE))
            if duration > len(attack) + len(decay):
                sustain = np.full(round(duration * SAMPLE_RATE - len(decay) - len(attack)), val_sustain)
            else:
                sustain = []
            release = np.linspace(val_sustain, 0, round(val_release*SAMPLE_RATE))
            return np.concatenate((attack, decay, sustain, release))

        if self.mode == 'ad':
            val_attack, val_decay = self.envelope
            attack = np.linspace(0, 1, round(val_attack*SAMPLE_RATE))
            decay = np.linspace(1, 0, round(val_decay*SAMPLE_RATE))
            return np.concatenate((attack, decay))

    def plot(self, ax=None) -> None:
        if not ax:
            fig, ax = plt.subplots()
        env = self.get_envelope(ENV_LENGTH_FOR_PLOTS)
        ax.plot(np.linspace(0, ENV_LENGTH_FOR_PLOTS, len(env)), env, 'black')

class SynthBase:

    def __init__(self, name: str, envelope: Envelope,  oscillators: list[Oscillator], osc_weights=None, volume = 1):
        self.name = name
        self.envelope = envelope
        self.oscillators = oscillators
        if not osc_weights:
            self.weights = list(np.full(len(oscillators), 1/len(oscillators)))
        else:
            assert len(oscillators) == len(osc_weights), 'osc_weights must contain weights for every oscillator'
            self.weights = list((osc_weights)/np.sum(osc_weights))
        self.volume = volume

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        dict = {
            'name': self.name,
            'envelope': self.envelope.to_dict(),
            'oscillators': [osc.to_dict() for osc in self.oscillators],
            'osc_weights': self.weights,
            'volume': self.volume
        }
        return dict
    
    def save_json(self, path) -> None:
        with open(path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    def from_dict(dict):
        envelope = Envelope(**dict.pop('envelope'))
        oscillators = [Oscillator(**osc) for osc in dict.pop('oscillators')]
        return(SynthBase(dict.pop('name'), envelope, oscillators, **dict))

    def from_json(path: str):
        with open(path) as file:
            return SynthBase.from_dict(json.load(file))
        
    def play_freq(self, length, freq, vel = 64) -> np.ndarray:
        envelope = self.envelope.get_envelope(length)
        duration = len(envelope)/SAMPLE_RATE
        arr = self.oscillators[0].play_freq(duration, freq) * self.weights[0]
        for osc, weight in zip(self.oscillators[1:], self.weights[1:]):
            arr += osc.play_freq(duration, freq) * weight
        return np.multiply(arr*(self.volume*(vel/127)), envelope)
    
    def play_note(self, length, note, vel = 64) -> np.ndarray:
        freq = TUNING*np.power(2, (note-69)/12)
        return self.play_freq(length, freq, vel)

    def play_noenv(self, lenght, freq) -> np.ndarray:
        arr = self.oscillators[0].play_freq(lenght, freq) * self.weights[0]
        for osc, weight in zip(self.oscillators[1:], self.weights):
            arr += osc.play_freq(lenght, freq) * weight
        return arr

    def set_vol(self, vol) -> None:
        self.volume = vol
    
    def set_weights(self, weights: list) -> None:
        assert len(weights) == len(self.oscillators), 'list must contain weights for every oscillator'
        self.weights = np.array(weights)/np.sum(weights)
    
    def get_fft(self, base_freq = FREQ_FOR_PLOTS) -> tuple[np.ndarray]:
        y = self.play_noenv(CYCLES_FOR_FFT/ base_freq, base_freq)
        N = len(y)
        T = 1.0 / SAMPLE_RATE
        yf = fft(y)
        xf = fftfreq(N, T)[:N//2]
        return xf, 2.0/N * np.abs(yf[0:N//2])

    def plot(self):
        fig = plt.figure()

        gs1 = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])

        self.envelope.plot(ax1)
        ax1.set_title('Envelope')
        ax1.get_yaxis().set_visible(False)

        ax2.plot(*self.get_fft())
        ax2.set_title(f'Fourier Transfrom \n input frequency = {FREQ_FOR_PLOTS} Hz')
        ax2.set_xscale('log')
        ax2.get_yaxis().set_visible(False)
        ax2.set_xlim(20, 20_000)

        gs1.tight_layout(fig, rect=[None, .6, None, None])

        gs2 = fig.add_gridspec(2, 1)
        ax3 = fig.add_subplot(gs2[0])
        ax4 = fig.add_subplot(gs2[1])

        for osc in self.oscillators:
            osc.plot(ax3)
        ax3.legend([f'{osc.mode}, w = {weight}' for osc, weight in zip(self.oscillators, self.weights)], loc='upper left')
        ax3.set_title(f'Oscillators ({CYCLES_FOR_PLOTS} cycles)')
        ax3.get_xaxis().set_visible(False)

        ax4.plot(self.play_noenv(CYCLES_FOR_PLOTS/FREQ_FOR_PLOTS, FREQ_FOR_PLOTS), 'black')
        ax4.set_title(f'Sum of Oscillators ({CYCLES_FOR_PLOTS} cycles)')
        ax4.get_xaxis().set_visible(False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gs2.tight_layout(fig, rect=[None, None, None, .6])

        left = min(gs1.left, gs2.left)
        right = max(gs1.right, gs2.right)

        gs1.update(left=left, right=right)
        gs2.update(left=left, right=right)
        return fig

class Effect:
    modes = ['band_pass', 'medfilt']
    def __init__(self, controls: dict, mode: str, on = True) -> None:
        assert mode in self.modes, f'unknown mode, try: {self.modes}'
        self.controls = controls
        self.mode = mode
        self.on = True
        if mode == 'band_pass':
            self.func = eff.band_pass
        if mode == 'medfilt':
            self.func = eff.medfilt
        try:
            self.apply(np.ndarray([1, 0, 1, 0, 1, 0.5, 1]))
        except TypeError:
            print('wrong keyword')


    def to_dict(self) -> dict:
        dict = {
            'controls': self.controls,
            'mode': self.mode
        }
        return dict
    

    def turn_off(self):
        self.on = False
    
    def turn_on(self):
        self.on = True

    def set_controls(self, **controls) -> None:
        self.controls = controls

    def apply(self, arr: np.ndarray) -> np.ndarray:
        if self.on:
            return self.func(arr, **self.controls)
        else:
            return arr



class Instrument(SynthBase):
    def __new__(cls, synth_base: SynthBase, effects: list[Effect]):
        synth_base.__class__ = cls
        return synth_base

    def __init__(self, synth_base: SynthBase, effects: list[Effect]) -> None:
        self.effects = effects
    
    def to_dict(self) -> dict:
        dict = super().to_dict()
        dict['effects'] = [eff.to_dict() for eff in self.effects]
        return dict

    def from_dict(dict):
        effcts = dict.pop('effects')
        effects = [Effect(**eff) for eff in effcts]
        base = SynthBase.from_dict(dict)
        return Instrument(base, effects)

    def save_json(self, dir_path: str) -> None:
        return super().save_json(dir_path)
    
    def from_json(path: str):
        if path == 'instruments/drums.json': #TODO nicely
            return Drums()
        with open(path) as file:
            return Instrument.from_dict(json.load(file))

    def play_freq(self, length, freq, vel = 64) -> np.ndarray:
        arr = super().play_freq(length, freq, vel)
        for eff in self.effects:
            arr = eff.apply(arr)
        return arr

    def plot(self):
        return super().plot()
    
    def play_noenv(self, lenght, freq) -> np.ndarray:
        arr = super().play_noenv(lenght, freq)
        for eff in self.effects:
            arr = eff.apply(arr)
        return arr

class Drums:
    def __init__(self) -> None:
        self.kick = load_wav('./samples/kick.WAV', 32767)*5
        self.snare = load_wav('./samples/snare.WAV', 32767)*5
        self.hihat = load_wav('./samples/hihat.WAV', 32767)*2
    
    def play_note(self, length, note , vel=64):
        if note == 36:
            return self.kick
        if note == 38:
            return self.snare
        if note == 42:
            return self.hihat
        print('undefined sample')
