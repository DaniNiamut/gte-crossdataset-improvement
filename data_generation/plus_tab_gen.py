import os
import json
import torch
import librosa
from tqdm import tqdm
from utils.misc import *
from utils.utils import midi_note_mode

class MonophonicTablatureEstimation():
    '''
    This class performs monophonic tablature estimation for audio of a guitar being played where only
    one guitar string is being plucked throughout the entire recording. This is used to provide tablature
    for the guitar sources in PLUS. This class is only fit for audio of a chromatic scale being played 
    on a single string at a fixed tempo.
    '''
    def __init__(self, frame_length_yin=2048, hop_length_yin=None, 
                 sr=None, padding_onset=0.05, len_of_silence_in_beats=1, 
                 len_of_note_in_beats=3, fmin=50, fmax=1000, chromatic_scale=True,
                 hop_length_fft=256, win_length_fft=512, n_fft=512,
                 open_string_midi_vals=MIDI_NOTE_NUMBERS):
        self.frame_length_yin = frame_length_yin
        if hop_length_yin:
            self.hop_length_yin = hop_length_yin
        else:
            self.hop_length_yin = self.frame_length_yin // 4
        self.padding_onset = padding_onset
        self.len_of_silence_in_beats= len_of_silence_in_beats
        self.len_of_note_in_beats = len_of_note_in_beats
        self.fmin = fmin
        self.fmax = fmax
        self.sr = sr
        self.hop_length_fft = hop_length_fft
        self.win_length_fft = win_length_fft
        self.n_fft = n_fft
        self.pick_f0_method = midi_note_mode
        self.events_recorded = 0
        self.total_events_to_record = 0
        self.warnings = []
        self.open_string_midi_vals = open_string_midi_vals
        self.chromatic_scale = chromatic_scale

    def load_plus(self, num_sources=None):
        '''
        num_sources: int > 0 or None
        num_sources states how many sources from the CHROM_SCALES_dataset will be loaded. If None,
        all sources are used.
        '''
        file_annotations = {}
        sets_of_guitar_sources = os.listdir(CHROM_SCALES_PATH + '/items')
        if num_sources:
            sets_of_guitar_sources = sets_of_guitar_sources[:num_sources]
        for set_gs in tqdm(sets_of_guitar_sources):
            set_gs_path = CHROM_SCALES_PATH + '/items/' + set_gs
            guitar_sources = os.listdir(set_gs_path)
            name_file_to_string = self._load_metadata(set_gs_path, dict_as_midi=False)[-1]
            guitar_sources.remove('metadata.json')
            file_annotations[set_gs] = {}

            for guitar_source in guitar_sources:
                guitar_source_path = set_gs_path + '/' + guitar_source
                string = name_file_to_string[guitar_source]
                onsets, offsets, strings, pitches = self.load_guitar_source(set_gs_path,
                                                                            guitar_source)
                # turn onset and offset times to actual time and not samples.
                # consider saving annotation to JAMS and saving paths for both.
                annotation = {'onset_time':onsets,
                            'offset_time':offsets,
                            'string':strings,
                            'pitch':pitches,}
                file_annotations[set_gs][string] = guitar_source_path, annotation
        return file_annotations

    def annotations(self, guitar_source, onsets, offsets,
                    pitches, max_fret, name_file_to_midi,):
        string_midi_val = name_file_to_midi[guitar_source]
        string_val = self.open_string_midi_vals.index(string_midi_val)
        strings = [string_val] * len(pitches)
        frets = [pitch - string_midi_val for pitch in pitches]

        if self.chromatic_scale:
            onsets, offsets, strings, pitches = self._handle_anomalies(guitar_source, onsets,
                                                                       offsets, strings,
                                                                       pitches, frets, max_fret)
        self.total_events_to_record += max_fret + 1
        self.events_recorded += len(onsets)

        return onsets, offsets, strings, pitches

    def midi_per_for_all_onsets(self, onsets, wav, 
                         sr, samples_per_beat):
        '''
        '''
        sample_nr_to_yin_frame = lambda sample_nr: int(sample_nr / int(self.hop_length_yin) + 1)
        pitches = []
        yin_frames = librosa.yin(wav, fmin=self.fmin, fmax=self.fmax, sr=sr,
                            frame_length=self.frame_length_yin, hop_length=self.hop_length_yin)
        for onset_idx in range(0, len(onsets)):
            onset = onsets[onset_idx]
            yin_frame_onset = sample_nr_to_yin_frame(onset)
            yin_frame_offset = 2 * sample_nr_to_yin_frame(samples_per_beat) + yin_frame_onset
            f0s_within_onset = yin_frames[yin_frame_onset:yin_frame_offset]
            f0 = self.pick_f0_method(list(f0s_within_onset))
            midi_pitch = round(librosa.hz_to_midi(f0))
            pitches.append(midi_pitch)
        return pitches
                
    def _refine_idx(self, onset_idx,
                 samples_per_beat, wav_len, padding=0):
        onset_idx = int(max(onset_idx - padding, 0))
        unrefined_offset_idx = (onset_idx + samples_per_beat 
                                    * self.len_of_note_in_beats)
        offset_idx = int(min(unrefined_offset_idx, wav_len - 1))
        return onset_idx, offset_idx
    
    def onset_for_all_regions(self, onset_regions, samples_per_beat, wav, sr):
        onsets, offsets = [], []
        
        for onset_region_idx, curr_onset_region in enumerate(onset_regions):
            padding_onset = int(self.padding_onset * sr)
            (curr_onset_region,
                curr_offset_region) = self._refine_idx(curr_onset_region, samples_per_beat, 
                                                       len(wav), padding_onset)
            curr_pred_onset = curr_onset_region + self.find_onset(
                                            wav[curr_onset_region:curr_offset_region])
            
            # Magic number to make predicted onset earlier:
            # In principle this is unrelated to the padding number in _refine_idx, 
            # but they happen to be the same value in this implementation.
            curr_pred_onset = curr_pred_onset - int(0.05 * sr)
            (curr_pred_onset,
                curr_pred_offset) = self._refine_idx(curr_pred_onset,
                                                samples_per_beat, len(wav))
            if (onset_region_idx != 0):
                prev_offset = offsets[-1]
                if curr_pred_onset < prev_offset:
                    onsets[-1] = prev_onset_region
                    offsets[-1] = prev_offset_region
                    curr_pred_onset = curr_onset_region
                    curr_pred_offset = curr_offset_region
                    # Deprecrated warning.
                    warning_str = ('There was an overlap in predicted onset and previous offset.' +
                    'The two overlapping onset intervals have been replaced with their' + 
                    'corresponding onset region intervals instead.')
                
            prev_onset_region = curr_onset_region
            prev_offset_region = curr_offset_region
            onsets.append(curr_pred_onset)
            offsets.append(curr_pred_offset)
        return onsets, offsets
                    
    def find_onset(self, wav):
        hann_window = torch.hann_window(self.win_length_fft, dtype=torch.float64)
        wav = torch.tensor(wav, dtype=torch.float64)
        win_len_half = int(round(self.win_length_fft / 2))
        wav = torch.concatenate((torch.zeros(win_len_half, dtype=torch.float64), wav, 
                                torch.zeros(win_len_half, dtype=torch.float64)), axis=0)
        abs_stft = torch.stft(wav, self.n_fft, self.hop_length_fft, self.win_length_fft, 
                              window=hann_window, return_complex=True, center=False, 
                              normalized=False,).abs()
        log_stft = torch.log(1 + 10 * abs_stft)
        stft_diff = log_stft[:,:-1] - log_stft[:,1:]
        stft_diff[stft_diff < 0] = 0
        novelty = torch.sum(stft_diff, axis=0)
        ind_max = torch.argmax(novelty)
        return ind_max * self.hop_length_fft
            
    def _load_metadata(self, set_gs_path, dict_as_midi=True):
        metadata_path = set_gs_path + '/' + 'metadata.json'
        with open(metadata_path, 'r') as path:
            metadata = json.load(path)
        tempo, max_fret = metadata['tempo'], metadata['highestFret']
        name_file_to_string = {metadata['recordings'][i+1]:metadata['recordings'][i] 
                                    for i in range(0, len(metadata['recordings']), 2)}
        if dict_as_midi:
            name_file_to_string = {key : STRING_NAME_TO_MIDI[name_file_to_string[key]]
                                    for key in name_file_to_string.keys()}
        return tempo, max_fret, name_file_to_string

    def load_guitar_source(self, set_gs_path, guitar_source):
        tempo, max_fret, name_file_to_midi = self._load_metadata(set_gs_path)
        guitar_source_path = set_gs_path + '/' + guitar_source
        wav, sr = librosa.load(guitar_source_path, sr=self.sr)
        samples_to_time = lambda sample: sample / sr 

        ## beat related constants
        samples_per_beat = sr / (tempo / 60)
        total_beats_in_wav = round(len(wav) / samples_per_beat)
        onset_regions = [samples_per_beat * i 
                        for i in range(0, total_beats_in_wav, 
                        self.len_of_silence_in_beats + self.len_of_note_in_beats)]
        
        onsets, offsets = self.onset_for_all_regions(onset_regions,
                                                samples_per_beat, wav, sr)
        pitches = self.midi_per_for_all_onsets(onsets, wav, 
                    sr, samples_per_beat)
        onsets, offsets, strings, pitches  = self.annotations(guitar_source, 
                                                    onsets, offsets, pitches, 
                                                    max_fret, name_file_to_midi,)
        onsets, offsets = map(samples_to_time, onsets), map(samples_to_time, offsets)
        onsets, offsets = list(onsets), list(offsets)
        return onsets, offsets, strings, pitches
    
    def _handle_anomalies(self, guitar_source, onsets, offsets, strings, pitches, frets, max_fret):
        warning_str = None
        expected_frets = [0]
        indices_to_remove, frets_seen = [], []
        frets_to_see = list(range(0, max_fret + 1))
        for onset_idx, fret in enumerate(frets):
            if fret not in expected_frets:
                indices_to_remove.append(onset_idx)
                if fret in frets_seen:
                    warning_str = (f'[{guitar_source}] Some frets are plucked more than once.')
                elif onset_idx != 0 and fret != frets[onset_idx - 1] + 1:
                    warning_str = (f'[{guitar_source}] The frets are not being plucked in order.')
                expected_frets.append(expected_frets[-1] + 1)
            else:
                expected_frets = [fret + 1]
                frets_seen.append(fret)

        if not warning_str and set(frets_seen) != set(frets_to_see):
            warning_str = (f'[{guitar_source}] {len(set(frets_seen)) - len(set(frets_to_see))}' +
                           ' frets are missing.')
        if warning_str:
            self.warnings.append(warning_str)
        for index_to_remove in sorted(indices_to_remove, reverse=True):
            onsets.pop(index_to_remove)
            offsets.pop(index_to_remove)
            strings.pop(index_to_remove)
            pitches.pop(index_to_remove)
            frets.pop(index_to_remove)
        return onsets, offsets, strings, pitches
