import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fftfreq, fft
from scipy import signal

from . import effects as eff_func
from . import oscillators as osc
from . import array_func as af

from .configs import CYCLES_FOR_PLOTS, DURATION_FOR_FFT, FREQ_FOR_PLOTS, INST_PATH, SAMPLE_RATE, ENV_LENGTH_FOR_PLOTS, TUNING

def load_wav(file_path: str):
    samplerate, arr = wavfile.read(file_path)
    assert samplerate == SAMPLE_RATE
    max_val = np.iinfo(arr.dtype).max
    arr = arr[:,0]/max_val
    return arr

def get_default_time_arr(duration):
    return np.linspace(0, duration, round(SAMPLE_RATE*duration))

class TimeKeeper:
    def __init__(self, cables: dict, envelope, **kwargs):
        """
        envelope is allways mapped to amplitude during playing of the note and is a list of floats
        envelope2 is can be mapped to frequency of the played note
        lfo is a dict containing 'mode', 'freq', 'modulation'
        cable keys[lfo_to_freq, env_to_freq, env2_to_freq]
        """
        self.cables = cables
        self.envelope = envelope
        if 'envelope2' in kwargs:
            self.envelope2 = kwargs.pop('envelope2')
        else:
            self.envelope2 = [.1]
        if 'lfo' in kwargs:
            self.lfo = kwargs.pop('lfo')
        else:
            self.lfo = {
                'mode': 'sine',
                'freq': 1,
                'modulation': 0
            }

    @staticmethod
    def make_envelope(envelope: list, duration: float) -> np.ndarray:
        if len(envelope) == 1:
            dur_attack, = envelope
            attack = np.linspace(1, 0, round(dur_attack*SAMPLE_RATE))
            return af.cut_or_pad(round(duration*SAMPLE_RATE), attack, pad_val=1)

        if len(envelope) == 2:
            dur_attack, dur_decay = envelope
            attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
            decay = np.linspace(1, 0, round(dur_decay*SAMPLE_RATE))
            return np.concatenate((attack, decay))

        if len(envelope) == 3:
            dur_attack, sus_decay, dur_release = envelope
            attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
            if round(duration*SAMPLE_RATE) > len(attack):
                length = (duration - len(attack)/SAMPLE_RATE)
                arr = get_default_time_arr(length)
                sustain = np.exp(np.log(sus_decay)*arr)
                last_val = sustain[-1]
            else:
                sustain = []
                last_val = 1
            release = np.linspace(last_val, 0, round(dur_release*SAMPLE_RATE))
            return np.concatenate((attack, sustain, release))


        if len(envelope) == 4:
            dur_attack, dur_decay, val_sustain, dur_release = envelope
            attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
            decay = np.linspace(1, val_sustain, round(dur_decay*SAMPLE_RATE))
            if round(duration*SAMPLE_RATE) > len(attack) + len(decay):
                sustain = np.full(round(duration * SAMPLE_RATE - len(decay) - len(attack)), val_sustain)
            else:
                sustain = []
            release = np.linspace(val_sustain, 0, round(dur_release*SAMPLE_RATE))
            return np.concatenate((attack, decay, sustain, release))

        if len(envelope) == 5:
            dur_attack, dur_decay, val_sustain, sus_decay, dur_release = envelope
            attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
            decay = np.linspace(1, val_sustain, round(dur_decay*SAMPLE_RATE))
            if round(duration*SAMPLE_RATE) > len(attack) + len(decay):
                length = (duration - (len(decay) + len(attack))/SAMPLE_RATE)
                arr = get_default_time_arr(length)
                sustain = np.exp(np.log(sus_decay)*arr)*val_sustain
                last_val = sustain[-1]
            else:
                sustain = []
                last_val = val_sustain
            release = np.linspace(last_val, 0, round(dur_release*SAMPLE_RATE))
            return np.concatenate((attack, decay, sustain, release))

    def __str__(self) -> str:
        return 'TimeKeeper:'

    def get_envelope(self, duration: float) -> np.ndarray:
        return self.make_envelope(self.envelope, duration)

    def get_time_array(self, length: float, duration: float):
        time = get_default_time_arr(duration)
        time += osc.get_stammfunc(self.lfo['mode'])(time*self.lfo['freq'] * 2*np.pi, self.lfo['modulation']) * self.cables['lfo_to_freq']
        time += self.make_envelope(self.envelope, length) * self.cables['env_to_freq']
        time += af.cut_or_pad(len(time), self.make_envelope(self.envelope2, length) * self.cables['env2_to_freq'])
        return time

    def to_dict(self) -> dict:
        dictionary = {
            'cables': self.cables,
            'envelope': self.envelope,
            'envelope2': self.envelope2,
            'lfo': self.lfo
        }
        return dictionary

    def plot_env(self, ax):
        env = self.get_envelope(ENV_LENGTH_FOR_PLOTS)
        ax.plot(np.linspace(0, ENV_LENGTH_FOR_PLOTS, len(env)), env, 'black')

class Oscillator:
    def __init__(self, mode: str, modulation = 0, detune_oct = 0, detune_cent = 0) -> None:
        self.mode = mode
        self.modulation = modulation
        self.cents = detune_cent
        self.octave = detune_oct
        self.func = getattr(osc, mode)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        dictionary = {
            'mode': self.mode,
            'modulation': self.modulation,
            'detune_oct': self.octave,
            'detune_cent': self.cents

        }
        return dictionary

    def play_freq(self, freq: float, time_arr: np.ndarray) -> np.ndarray:
        freq = (freq * 2**self.octave)  * 1.0005777895065548**self.cents
        wave = self.func(time_arr*freq * 2*np.pi, self.modulation)
        return wave

    def plot(self, ax = None) -> None:
        if not ax:
            _, ax = plt.subplots()
        duration = CYCLES_FOR_PLOTS/FREQ_FOR_PLOTS
        time_arr = get_default_time_arr(duration)
        ax.plot(self.play_freq(FREQ_FOR_PLOTS, time_arr))
        ax.set_label(self.mode)

class Effect:
    def __init__(self, mode: str, controls: dict, on = True) -> None:
        self.mode = mode
        self.controls = controls
        self.on = on
        self.func = getattr(eff_func, mode)

    def __str__(self):
        controls = ''
        for key, val in self.controls.items():
            controls += f'{key} = {val}, '
        return f'Effect("{self.mode}", {controls})'

    def to_dict(self) -> dict:
        dictionary = {
            'mode': self.mode,
            'controls': self.controls
        }
        return dictionary

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

class Synthesizer:
    def __init__(self, effects: list[Effect], name: str, time_keeper: TimeKeeper,  oscillators: list[Oscillator], osc_weights=None, volume = 1) -> None:
        self.name = name
        self.volume = volume
        self.time_keeper = time_keeper
        self.oscillators = oscillators
        if not osc_weights:
            self.weights = list(np.full(len(oscillators), 1/len(oscillators)))
        else:
            assert len(oscillators) == len(osc_weights), 'osc_weights must contain weights for every oscillator'
            self.weights = list((osc_weights)/np.sum(osc_weights))
        self.effects = effects

    def __str__(self):
        return f'Synthesizer: {self.name}'

    def to_dict(self) -> dict:
        dictionary = {
            'name': self.name,
            'volume': self.volume,
            'time_keeper': self.time_keeper.to_dict(),
            'oscillators': [osc.to_dict() for osc in self.oscillators],
            'osc_weights': self.weights,
            'effects': [eff.to_dict() for eff in self.effects]
        }
        return dictionary

    @staticmethod
    def from_dict(dictionary):
        effects = [Effect(**eff) for eff in dictionary.pop('effects')]
        envelope = TimeKeeper(**dictionary.pop('time_keeper'))
        oscillators = [Oscillator(**osc) for osc in dictionary.pop('oscillators')]
        return Synthesizer(effects, dictionary.pop('name'), envelope, oscillators, **dictionary)

    def save_json(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def from_json(path: str):
        with open(path, encoding='utf-8') as file:
            return Synthesizer.from_dict(json.load(file))

    def play_freq(self, length, freq, vel = 64) -> np.ndarray:
        envelope = self.time_keeper.get_envelope(length)
        duration = len(envelope)/SAMPLE_RATE
        time_arr = self.time_keeper.get_time_array(length, duration)
        arr = np.zeros(len(envelope))
        for oscillator, weight in zip(self.oscillators, self.weights):
            arr += oscillator.play_freq(freq, time_arr) * weight
        arr = np.multiply(arr*(self.volume*(vel/127)), envelope)
        for eff in self.effects:
            arr = eff.apply(arr)
        return arr

    def set_weights(self, weights: list) -> None:
        assert len(weights) == len(self.oscillators), 'list must contain weights for every oscillator'
        self.weights = np.array(weights)/np.sum(weights)

    def play_note(self, length, note, vel = 64) -> np.ndarray:
        freq = TUNING*np.power(2, (note-69)/12)
        return self.play_freq(length, freq, vel)

    def get_fft(self, base_freq = FREQ_FOR_PLOTS) -> tuple[np.ndarray]:
        y = self.play_freq(DURATION_FOR_FFT, base_freq)
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

        self.time_keeper.plot_env(ax1)
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

        for oscillator in self.oscillators:
            oscillator.plot(ax3)
        ax3.legend([f'{osc.mode}, w = {weight}' for osc, weight in zip(self.oscillators, self.weights)], loc='upper left')
        ax3.set_title(f'Oscillators ({CYCLES_FOR_PLOTS} cycles)')
        ax3.get_xaxis().set_visible(False)

        duration = CYCLES_FOR_PLOTS/FREQ_FOR_PLOTS
        time_arr = get_default_time_arr(duration)
        arr = np.zeros(round(duration*SAMPLE_RATE))
        for weight, oscillator in zip(self.weights, self.oscillators):
            arr += oscillator.play_freq(FREQ_FOR_PLOTS, time_arr)*weight
        ax4.plot(arr, 'black')
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

class Drums:
    def __init__(self) -> None:
        self.name = 'drums'
        self.volume = 1
        self.kick = load_wav('./samples/kick.wav')*5
        self.snare = load_wav('./samples/snare.wav')*5
        self.hihat = load_wav('./samples/hihat.wav')*.8
        self.hihat_half = load_wav('./samples/hihat_half.wav')*2

    def __str__(self) -> str:
        return 'Drums'

    def play_note(self, length, note , vel=64):
        if note == 36:
            return self.kick * self.volume * vel/127
        if note == 38:
            return self.snare * self.volume * vel/127
        if note == 42:
            return self.hihat * self.volume * vel/127
        if note == 46:
            return (self.hihat_half * self.volume * vel/127)[:round(length*SAMPLE_RATE)]
        raise Exception('note not found')

def get_user_inst(voice_name = ''):
    while True:
        name = input(f'filename for instrument in voice "{voice_name}" : ')
        try:
            if name == 'drums':
                return Drums()
            return Synthesizer.from_json(f'{INST_PATH}{name}.json')
        except FileNotFoundError:
            print('No such file.')

def inst_from_name(name: str):
    if name == 'drums':
        return Drums()
    path = f'{INST_PATH}{name}.json'
    return Synthesizer.from_json(path)
