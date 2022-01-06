import numpy as np

from .configs import SAMPLE_RATE
from . import array_func as af


def make(envelope: list, sus_length: float) -> np.ndarray:
    if len(envelope) == 1:
        dur_decay, = envelope
        return np.linspace(1, 0, round(dur_decay*SAMPLE_RATE))

    if len(envelope) == 2:
        dur_attack, dur_decay = envelope
        attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
        decay = np.linspace(1, 0, round(dur_decay*SAMPLE_RATE))
        return np.concatenate((attack, decay))

    if len(envelope) == 3:
        dur_attack, sus_half, dur_release = envelope
        attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
        if round(sus_length*SAMPLE_RATE) > len(attack):
            length = (sus_length - len(attack)/SAMPLE_RATE)
            arr = af.get_default_time_arr(length)
            sustain = np.exp(arr * (-0.6931471806/sus_half))
        else:
            sustain = [1]
        release = np.linspace(sustain[-1], 0, round(dur_release*SAMPLE_RATE))
        return np.concatenate((attack, sustain, release))


    if len(envelope) == 4:
        dur_attack, dur_decay, val_sustain, dur_release = envelope
        attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
        decay = np.linspace(1, val_sustain, round(dur_decay*SAMPLE_RATE))
        if round(sus_length*SAMPLE_RATE) > len(attack) + len(decay):
            sustain = np.full(round(sus_length * SAMPLE_RATE - len(decay) - len(attack)), val_sustain)
        else:
            sustain = [val_sustain]
        release = np.linspace(val_sustain, 0, round(dur_release*SAMPLE_RATE))
        return np.concatenate((attack, decay, sustain, release))

    if len(envelope) == 5:
        dur_attack, dur_decay, val_sustain, sus_half, dur_release = envelope
        attack = np.linspace(0, 1, round(dur_attack*SAMPLE_RATE))
        decay = np.linspace(1, val_sustain, round(dur_decay*SAMPLE_RATE))
        if round(sus_length*SAMPLE_RATE) > len(attack) + len(decay):
            length = (sus_length - (len(decay) + len(attack))/SAMPLE_RATE)
            arr = af.get_default_time_arr(length)
            sustain = np.exp(arr * (-0.6931471806/sus_half))*val_sustain
        else:
            sustain = [val_sustain]
        release = np.linspace(sustain[-1], 0, round(dur_release*SAMPLE_RATE))
        return np.concatenate((attack, decay, sustain, release))

def _linear_stamm(duration, start, stop, start_val=0):
    time_arr = af.get_default_time_arr(duration)
    return start*time_arr + time_arr**2*(stop-start)/(2*duration) + start_val

def stamm(envelope: list, sus_length: float):
    if len(envelope) == 1:
        dur_decay, = envelope
        return _linear_stamm(dur_decay, 1, 0)

    if len(envelope) == 2:
        dur_attack, dur_decay = envelope
        attack = _linear_stamm(dur_attack, 0, 1)
        decay = _linear_stamm(dur_decay, 1, 0, attack[-1])
        return np.concatenate((attack, decay))

    if len(envelope) == 3:
        dur_attack, sus_half, dur_release = envelope
        attack = _linear_stamm(dur_attack, 0, 1)
        if round(sus_length*SAMPLE_RATE) > len(attack):
            length = (sus_length - len(attack)/SAMPLE_RATE)
            arr = af.get_default_time_arr(length)
            sustain = np.exp(arr * (-0.6931471806/sus_half))*(-sus_half/0.6931471806) + attack[-1] + sus_half/0.6931471806
        else:
            sustain = [attack[-1]]
        last_val = np.exp(length*(-0.6931471806/sus_half))
        release = _linear_stamm(dur_release, last_val, 0, sustain[-1])
        return np.concatenate((attack, sustain, release))


    if len(envelope) == 4:
        dur_attack, dur_decay, val_sustain, dur_release = envelope
        attack = _linear_stamm(dur_attack, 0, 1)
        decay = _linear_stamm(dur_decay, 1, val_sustain, attack[-1])
        if round(sus_length*SAMPLE_RATE) > len(attack) + len(decay):
            sustain = _linear_stamm(sus_length - (len(attack)+len(decay))/SAMPLE_RATE, val_sustain, val_sustain, decay[-1])
        else:
            sustain = [decay[-1]]
        release = _linear_stamm(dur_release, val_sustain, 0, sustain[-1])
        return np.concatenate((attack, decay, sustain, release))

    if len(envelope) == 5:
        dur_attack, dur_decay, val_sustain, sus_half, dur_release = envelope
        attack = _linear_stamm(dur_attack, 0, 1)
        decay = _linear_stamm(dur_decay, 1, val_sustain, attack[-1])
        length = (sus_length - (len(attack) + len(decay))/SAMPLE_RATE)
        if round(sus_length*SAMPLE_RATE) > len(attack) + len(decay):
            arr = af.get_default_time_arr(length)
            sustain = np.exp(arr * (-0.6931471806/sus_half)) * (-val_sustain*sus_half/0.6931471806) + decay[-1] + sus_half/0.6931471806
        else:
            sustain = [decay[-1]]
        last_val = np.exp(length*(-0.6931471806/sus_half)) * val_sustain
        release = _linear_stamm(dur_release, last_val, 0, sustain[-1])
        return np.concatenate((attack, decay, sustain, release))
