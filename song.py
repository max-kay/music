import numpy as np
import mido

from configs import SAMPLE_RATE
import instruments as inst

class Voice:
    def __init__(self, name: str, instrument: inst.Instrument, notes: list, ccs: list) -> None:
        self.name = name
        self.instrument = instrument
        self.notes = notes
        self.ccs = ccs

    @staticmethod
    def from_midi_track(track: mido.MidiTrack, ticks_per_second):
        current_time = 0
        current_notes = {}
        notes = []
        for msg in track:
            current_time += msg.time
            if msg.type == 'instrument_name':
                valid_name = False
                while not valid_name:
                    file_path = input(f'Filepath for Instrument in voice "{msg.name}" :    instruments/')
                    try:
                        instrument = inst.Instrument.from_json(f'instruments/{file_path}.json')
                        print('Instrument assigned')
                        valid_name = True
                    except FileNotFoundError:
                        print('No such file.')
                    name = f'{msg.name}'
            if msg.type == 'note_on':
                current_notes[msg.note] = (current_time, msg.velocity)
            if msg.type == 'note_off':
                if msg.note in current_notes:
                    onset, vel = current_notes.pop(msg.note)
                    notes.append([onset/ticks_per_second, ((current_time-onset)/ticks_per_second, msg.note, vel)])
        ccs = [[500000000, (0, 0, 0)]]
        return Voice(name, instrument, notes, ccs)

    def play(self):
        notes = self.notes.copy()
        ccs = self.ccs.copy()
        arr = np.zeros(SAMPLE_RATE) #TODO nicely
        note_onset, current_note = notes.pop(0)
        c_time, control_change = ccs.pop(0)
        while len(notes) > 0:
            if c_time <= note_onset:
                c_time, control_change = ccs.pop(0)
            else:
                note = self.instrument.play_note(*current_note)
                pos = round(note_onset*SAMPLE_RATE)
                arr = add_with_index(arr, note, pos)
                note_onset, current_note = notes.pop(0)
        note = self.instrument.play_note(*current_note)
        pos = round(note_onset*SAMPLE_RATE)
        arr = add_with_index(arr, note, pos)
        arr = np.trim_zeros(arr, 'b')
        return arr

def add_with_index(fix: np.ndarray, mobile: np.ndarray, index) -> np.ndarray:
    if len(mobile) + index > len(fix):
        fix = np.pad(fix, (0, len(mobile) + index - len(fix)), constant_values=(0, 0))
    for i, val in zip(range(index, index+len(mobile)), mobile):
        fix[i] += val
    return fix

def add_dif_len(arr1, arr2):
    if len(arr1) < len(arr2):
        arr1 = np.pad(arr1, (0, len(arr2)-len(arr1)), constant_values=(0, 0))
    else:
        arr2 = np.pad(arr2, (0, len(arr1)-len(arr2)), constant_values=(0, 0))
    return np.add(arr1, arr2)

class Song:
    def __init__(self, voices: list) -> None:
        self.voices = voices

    @staticmethod
    def from_midi(filepath):
        midi = mido.MidiFile(filepath)
        ticks_per_beat = midi.ticks_per_beat
        tracks = midi.tracks
        meta_track = tracks.pop(0)
        tempo = meta_track[3].tempo
        ticks_per_second = 1_000_000*ticks_per_beat//tempo
        voices = [Voice.from_midi_track(track, ticks_per_second) for track in tracks]
        return Song(voices)

    def play(self):
        arr = np.zeros(100)
        for voice in self.voices:
            arr = add_dif_len(voice.play(), arr)
        return np.trim_zeros(arr, 'b')
