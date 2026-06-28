import os
import re
import json
import torch
import random
import librosa
import soundfile
from tqdm import tqdm
from utils.misc import *
from utils.utils import error_handling_makedirs, get_audio_annot_folder_path, tukey_window
from data_handling.picker_functions import dataset_picker_dict
from data_handling.load import (fret_from_pitch, index_metadata_dict, 
                                save_dict_as_jams, file_grouper, annot_loader_dict)
'''
Does not necessarily have to be a class, reason being that this could wind up being used
with VAE. This is seeming less and less likely though.
'''

class AudioOverlay():
    
    def __init__(self, sr=44100, fade_alpha=0.1, cut_off_buffer=0.1,
                 string_names=CHROME_SCALES_STRING_NAMES):
        '''
        cut_off_buffer : in seconds (maybe we should change it to samples)
        '''
        self.sr = sr
        self.warnings = []
        self.alpha = fade_alpha
        self.buffer = cut_off_buffer
        self.time_to_sample = lambda time : int(round(time * self.sr))
        self.string_names = string_names
        self.string_num_to_name = {MIDI_NOTE_NUMBERS[i] : self.string_names[i] for i in range(6)}
        self.prev_source_paths = None
        self.wav_sources = None
        self.dataset_name = 'PLUS'
        pass

    def _index_metadata_dict(self, metadata_dict, note_idx, time_in_samples=False):
        onset_time, offset_time, string, fret = index_metadata_dict(metadata_dict, note_idx,
                                                                     pitch_key='fret')
        if time_in_samples:
            onset_time = self.time_to_sample(onset_time)
            offset_time = self.time_to_sample(offset_time)
        return onset_time, offset_time, string, fret

    def _find_where_note_played(self, annot, fret, string, note_idx, note_idx_to_remove):
        if 'fret' not in annot:
            annot = fret_from_pitch(annot)
        # Find where note is played
        try:
            ov_idx = annot['fret'].index(fret)
            return ov_idx, annot, note_idx_to_remove
        except ValueError:
            # Handle when note is not played.
            warning_str = (f'String {self.string_names[string]}, Fret {fret} not found,'
                + 'therefore it will be skipped.')
            self.warnings.append(warning_str)
            note_idx_to_remove.append(note_idx)
            return None, annot, note_idx_to_remove

    def overlay_audio(self, tabs_to_overlay, plus_source_dict, return_tabs=False):
        '''
        tabs_to_overlay : dict -> {'onset_time':[...], ...,'pitch':[...],}
        plus_source_dict : dict -> {'E':(path, {'onset_time':[...], ...,'pitch':[...],}), 
                                        'A':..., 'D':..., 'G':..., 'B':..., 'e':...,}
        duration : float
        sr : int
        '''
        note_idx_to_remove = []
        audio_duration = max(tabs_to_overlay['offset_time']) + 0.5
        output_tabs = tabs_to_overlay.copy()
        if 'fret' not in tabs_to_overlay:
            tabs_to_overlay = fret_from_pitch(tabs_to_overlay)
        # The '+1' is magic.
        wav = torch.zeros(int(audio_duration * self.sr) + 1)

        # Load audios
        source_paths = [plus_source_dict[string_name][0]
                        for string_name in self.string_names]
        if source_paths == self.prev_source_paths:
            wav_sources = self.wav_sources
        else:
            wav_sources = [librosa.load(path, sr=self.sr)[0] for path in source_paths]
            wav_sources = [torch.tensor(wav) for wav in wav_sources]
            self.wav_sources = wav_sources
            self.prev_source_paths = source_paths

        for note_idx in range(len(tabs_to_overlay['pitch'])):
            # Get information of note to overlay.
            metadata_vals = self._index_metadata_dict(tabs_to_overlay, note_idx,
                                                      time_in_samples=True)
            onset_time, offset_time, string, fret = metadata_vals
            duration = offset_time - onset_time + self.time_to_sample(self.buffer)

            # Get information of note which must be overlayed.
            annot = plus_source_dict[self.string_names[string]][1]
            wav_ov = wav_sources[string]
            ov_idx, annot, note_idx_to_remove = self._find_where_note_played(annot, fret,
                                                        string, note_idx,
                                                        note_idx_to_remove)
            # In case ov_idx does not exist
            if ov_idx is None:
                continue
            ov_onset_time, ov_offset_time = self._index_metadata_dict(annot, ov_idx,
                                                                time_in_samples=True)[:2]
            ov_duration = ov_offset_time - ov_onset_time

            # load specific note section
            chosen_duration = ov_duration if ov_duration <= duration else duration
            note_audio = wav_ov[ov_onset_time:ov_onset_time + chosen_duration]
            # taper edges of audio
            note_audio *= tukey_window(chosen_duration, alpha=self.alpha)
            wav[onset_time:onset_time + chosen_duration] += note_audio
            offsets = output_tabs['offset_time'].copy()
            offsets[note_idx] = chosen_duration + onset_time
        
        # Fix output_tabs
        for key in output_tabs:
            key_list = list(output_tabs[key]).copy()
            for note_idx in sorted(note_idx_to_remove, reverse=True):
                key_list.pop(note_idx)
            output_tabs[key] = key_list

        if return_tabs:
            return wav, output_tabs
        else:
            return wav
        
    # should plus be generated as random
    def gen_plus(self, dataset, file_annotations, from_sym_ds=False):
        '''
        This will overlay a random selected guitar source onto all tablature in a dataset.
        '''
        file_annotations_keys = list(file_annotations.keys())
        plus_path, annot_path, audio_path = get_audio_annot_folder_path(self.dataset_name)
        for path in [plus_path, annot_path, audio_path]:
            error_handling_makedirs(path)

        match_dict = file_grouper(dataset=dataset, path_format=True, return_match_dict=True)
        # function that returns a string
        annot_picker = dataset_picker_dict[dataset][0]
        # kwargs to annot_picker
        annot_kwargs = dataset_picker_dict[dataset][2]
        if from_sym_ds is not True:
            # function that returns a list with string(s).
            audio_picker = dataset_picker_dict[dataset][1]
            # kwargs to audio_picker
            audio_kwargs = dataset_picker_dict[dataset][3]

        for key in tqdm(match_dict.keys()):
            possible_annot = match_dict[key][0]
            possible_audio = match_dict[key][1]
            if len(possible_annot) == 0:
                continue
            if from_sym_ds is not True:
                if len(possible_audio) == 0:
                    continue
                possible_audio = audio_picker(possible_audio, **audio_kwargs)
                num_gen_audios = len(possible_audio)
            else:
                num_gen_audios = 1
            annot = annot_picker(possible_annot, **annot_kwargs)
            _, ext = os.path.splitext(annot)
            plus_sources = random.sample(file_annotations_keys, num_gen_audios)

            for plus_source_dict in plus_sources:
                tabs_to_overlay = annot_loader_dict[ext](annot)
                wav, tabs = self.overlay_audio(tabs_to_overlay,
                                            file_annotations[plus_source_dict],
                                            return_tabs=True)
                # Saving
                prefix = re.sub(r'\.[a-zA-Z0-9]{1,4}', '', key)
                file_name = f'/{prefix}_{plus_source_dict}'
                soundfile.write(audio_path + file_name + '.mp3', wav, self.sr)
                save_dict_as_jams(annot_path + file_name + '.jams', tabs)
