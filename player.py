import json
import numpy as np
import mido
from scipy.io import wavfile

from configs import INST_PATH, PLAYR_PATH, SAMPLE_RATE
import instruments as inst
import array_func as af

class Voice:
    def __init__(self, name: str, instrument: inst.Synthesizer, notes: list, ccs: list) -> None:
        self.name = name
        self.instrument = instrument
        self.notes = notes
        self.ccs = ccs

    def __str__(self) -> str:
        return f'Voice: {self.name} Instrument: {self.instrument.name} | {len(self.notes)} notes, {len(self.ccs)} control changes'

    @staticmethod
    def from_midi_track(track: mido.MidiTrack, ticks_per_second):
        current_time = 0
        current_notes = {}
        notes = []
        for msg in track:
            current_time += msg.time
            if msg.type == 'instrument_name':
                instrument = inst.get_user_inst(msg.name)
                name = f'{msg.name}'
            if msg.type == 'note_on':
                current_notes[msg.note] = (current_time, msg.velocity)
            if msg.type == 'note_off':
                if msg.note in current_notes:
                    onset, vel = current_notes.pop(msg.note)
                    notes.append([onset/ticks_per_second, ((current_time-onset)/ticks_per_second, msg.note, vel)])
        ccs = [[float('inf'), (0, 0, 0)]]
        return Voice(name, instrument, notes, ccs)

    def asign_inst(self):
        self.instrument = inst.get_user_inst(self.name)

    def to_dict(self):
        dictionary = {
            'name': self.name,
            'instrument_name': self.instrument.name,
            'notes': self.notes,
            'ccs': self.ccs
        }
        return dictionary

    @staticmethod
    def from_dict(dictionary: dict):
        dictionary['instrument'] = inst.inst_from_name(dictionary.pop('instrument_name'))
        return Voice(**dictionary)

    def play(self) -> np.ndarray:
        notes = self.notes.copy()
        ccs = self.ccs.copy()
        arr = np.zeros(SAMPLE_RATE)
        note_onset, current_note = notes.pop(0)
        c_time, control_change = ccs.pop(0)
        while len(notes) > 0:
            if c_time <= note_onset:
                c_time, control_change = ccs.pop(0)
            else:
                note = self.instrument.play_note(*current_note)
                pos = round(note_onset*SAMPLE_RATE)
                arr = af.add_with_index(arr, note, pos)
                note_onset, current_note = notes.pop(0)
        note = self.instrument.play_note(*current_note)
        pos = round(note_onset*SAMPLE_RATE)
        arr = af.add_with_index(arr, note, pos)
        arr = np.trim_zeros(arr, 'b')
        return arr

class Player:
    def __init__(self, name: str, voices: list, effects: list[inst.Effect] = []) -> None:
        self.name = name
        self.voices = voices
        self.effects = effects

    def __str__(self) -> str:
        string = f'Player: {self.name}\n  Voices:\n'
        for voice in self.voices:
            string += f'    {voice.__str__()}\n'
        string += ' Effects:\n'
        for effect in self.effects:
            string += f'    {effect.__str__()}\n'
        return string

    @staticmethod
    def from_midi(filepath):
        name = input('name for Player: ')
        midi = mido.MidiFile(filepath)
        ticks_per_beat = midi.ticks_per_beat
        tracks = midi.tracks
        meta_track = tracks.pop(0)
        tempo = meta_track[3].tempo
        ticks_per_second = 1_000_000*ticks_per_beat/tempo
        voices = [Voice.from_midi_track(track, ticks_per_second) for track in tracks]
        return Player(name, voices)

    def to_dict(self) -> dict:
        dictionary = {
            'name': self.name,
            'voices': [voice.to_dict() for voice in self.voices]
        }
        return dictionary

    def save_json(self):
        with open(f'{PLAYR_PATH}{self.name}.json', 'w', encoding='utf8') as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def from_dict(dictionary: dict):
        voices = [Voice.from_dict(voice) for voice in dictionary['voices']]
        return Player(dictionary.pop('name'), voices)

    @staticmethod
    def from_json(path: str):
        with open(path, encoding='utf8') as file:
            return Player.from_dict(json.load(file))

    def play(self) -> np.ndarray:
        arr = np.zeros(100)
        for voice in self.voices:
            arr = af.add_dif_len(voice.play(), arr)
        arr = np.trim_zeros(arr, 'b')
        for effect in self.effects:
            arr = effect.apply(arr)
        return arr

    def add_effect(self, effect: inst.Effect) -> None:
        self.effects.append(effect)

    def save_wav(self, filename = None, dtype = 'int32') -> None:
        if not filename:
            filename = f'./out/{self.name}.wav'
        arr = self.play()
        dtype = np.dtype(dtype)
        data = arr * np.iinfo(dtype).max *0.1
        data = data.astype(dtype)
        wavfile.write(filename, SAMPLE_RATE, data)
