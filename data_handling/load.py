import os
import re
import csv
import jams
import mido
import h5py
import torch
import librosa
import guitarpro
from tqdm import tqdm
import xml.etree.ElementTree as ET
from utils.misc import *
from data_handling.picker_functions import *
from utils.utils import get_audio_annot_folder_path, dict_from_lists

# The line below is necessary to account for JAM differences in the SynthTab annotation. The
# note_tab.json file it refers to has also been provided by SynthTab.
# See their repo: https://github.com/yongyizang/SynthTab/blob/main/gp_to_JAMS/guitarpro_utils.py
jams.schema.add_namespace(SYNTHTAB_JSON_PATH)

def standardize_keys_dict(metadata_dict, pitch_key, string_key, onset_key, 
                          offset_key, **kwargs):
    '''Makes key names the same because it is a bit tedious to 
        have to input different key names for different sources.

        The new keys will be:
        'pitch', 'string', 'onset_time', 'offset_time'.
        If kwargs is provided, the key should be a key in metadata_dict, 
        with the value in kwargs being the new name of the key in the dictionary. 
        If None is given for the key values, then no name change will occur.
        '''
    standardized_dict = metadata_dict.copy()
    old_keys = [pitch_key, string_key, onset_key, offset_key]
    new_keys = ['pitch', 'string', 'onset_time', 'offset_time']
    for idx in range(len(old_keys)):
        if old_keys[idx] is not None:
            standardized_dict[new_keys[idx]] = standardized_dict.pop(old_keys[idx])
    for key, value in kwargs.items():
        if key in standardized_dict:
            standardized_dict[value] = standardized_dict.pop(key)
    return standardized_dict

def ticks_to_sec(time_in_tick, tempos, tempo_onsets, 
                 quarternote_res=QUARTER_TIME):
    '''
    SynthTab jams and guitarpro files come with time in ticks instead of seconds. The conversion
    of ticks to seconds requires knowing the quarter notes per minute and the amount of ticks per
    second. An addition to this is that tempo can change with time, to take this into account the
    order and duration of each tempo must also be given.

    time_in_tick : list or an int. This is the tick value which must be converted into seconds.
    quarternote_res : Quarter note resolution.
    '''
    tempos = [tempo for _, tempo in sorted(zip(tempo_onsets, tempos))]
    tempo_onsets = sorted(tempo_onsets)
    if isinstance(time_in_tick, list):
        times_in_sec = [ticks_to_sec(time, tempos, tempo_onsets, quarternote_res) 
                for time in time_in_tick]
        return times_in_sec
    max_idx = max(idx for idx, onset in enumerate(tempo_onsets) if onset <= time_in_tick)
    prev_beats_passed = sum((tempo_onsets[idx + 1] - tempo_onsets[idx]) / 
                            (quarternote_res * tempos[idx])
                            for idx in range(max_idx))
    prev_min_passed = 0 if max_idx == 0 else (prev_beats_passed)
    curr_beats_passed = time_in_tick - tempo_onsets[max_idx]
    curr_min_passed = curr_beats_passed / (quarternote_res * tempos[max_idx])
    return 60 * (curr_min_passed + prev_min_passed)

def extend_frame_annot_to_list(frame_annot, keys_to_remove, expected_keys,
                               key_list, metadata_list, string):
    (keys_to_remove, key_list,  metadata_list) = (keys_to_remove.copy(), 
                                                  key_list.copy(),
                                                  metadata_list.copy())
    curr_key_list = list(frame_annot._fields)
    # curr_metadata_list is a list of values for the key above.
    curr_metadata_list = list(frame_annot)
    for idx, metadata in enumerate(curr_metadata_list):
        # sometimes the jams file has many subtrees, which comes as dictionaries. 
        # In this case we flatten it back to a list.
        if isinstance(metadata, dict):
            keys_to_remove.add(curr_key_list[idx])
            keys_to_add = list(metadata.keys())
            values_to_add = list(metadata.values())
            # Sometimes within the sub-dictionary a key value that is in the original
            # dictionary is repeated, so we remove those.
            keys_to_add = [elmnt for elmnt in keys_to_add if elmnt not in curr_key_list]
            values_to_add = [values_to_add[idx] for idx, elmnt in enumerate(keys_to_add)
                             if elmnt not in curr_key_list]
            curr_key_list += keys_to_add
            curr_metadata_list += values_to_add

    curr_key_list.append('string')
    curr_metadata_list.append(string)
    # Make sure the lists for each key will be of equal size even if a key value is not
    # specified for each event.
    for key in expected_keys:
        if key not in curr_key_list:
            curr_key_list.append(key)
            curr_metadata_list.append(None)

    # Add values and corresponding keys from this iteration.
    key_list += curr_key_list
    metadata_list += curr_metadata_list

    return (keys_to_remove, key_list,
             metadata_list)

def clean_metadict(metadata_dict, standardize_string_format=False, 
                   reverse_string=False, string_key='string', keys_to_remove=None):
    '''Sometimes the metadata_dict is not clean after creating it. 
    This function addresses when:
    1.  values in metadict are not a float or int.
    2.  string values are in format [1,2,3,4,5,6] or [5,4,3,2,1,0] instead of
    [0,1,2,3,4,5] where E2=0, A=1, D=2, G=3, B=4, E4=5.
    3.  Keys do not have values of equal length or keys_to_remove is given.
    Only used for load_midi_as_dict and load_xml_as_dict.
    '''
    if keys_to_remove is None:
        keys_to_remove = set()
    for key in metadata_dict.keys():
        if len(metadata_dict[key]) < len(metadata_dict[string_key]):
            keys_to_remove.add(key)
        try:
            float_vers_of_list = [float(metadata) for metadata in metadata_dict[key]]
            int_vers_of_list = [int(float_data) for float_data in float_vers_of_list]
            if int_vers_of_list == float_vers_of_list:
                metadata_dict[key] = int_vers_of_list
            else:
                metadata_dict[key] = float_vers_of_list
        except ValueError:
            continue
        except TypeError:
            continue
    for remove_key in keys_to_remove:
            metadata_dict.pop(remove_key)
    if standardize_string_format:
        metadata_dict[string_key] = [string_val - 1 
                                     for string_val in metadata_dict[string_key]]
    if reverse_string:
        metadata_dict[string_key] = [int(abs(string_val - 5)) 
                                     for string_val in metadata_dict[string_key]]
    return metadata_dict

def load_jam_as_dict(path):
    ticks_to_sec_change = False
    key_list, metadata_list, keys_to_remove = [], [], set()
    jam_file = jams.load(path)

    for string in range(6):

        annots = jam_file.annotations['note_midi'], jam_file.annotations['note_tab']
        if len(annots[0]) > len(annots[1]):
            # note_midi (Standard namespace)
            string_annot = annots[0][string]
            reverse_string = False
        elif len(annots[0]) < len(annots[1]):
            # note_tab (SynthTab)
            string_annot = annots[1][string]
            ticks_to_sec_change = True
            reverse_string = True
        else:
            raise KeyError('Unsure on whether to use note_midi or note_tab as namespace.')

        # Retrieve all keys that will be in metadata_dict.
        # We need this because sometimes events have missing keys.
        keys = [key for event in string_annot for key in event._fields]
        subkeys = [list(val.keys()) for event in string_annot 
                for val in event if isinstance(val, dict)]
        for sublist in subkeys:
            keys += sublist
        expected_keys = (set(keys))

        for frame_annot in string_annot.data:
            output = extend_frame_annot_to_list(frame_annot, keys_to_remove, expected_keys,
                                                key_list, metadata_list, string)
            keys_to_remove, key_list, metadata_list = output
    metadata_dict = dict_from_lists(key_list, metadata_list)

    metadata_dict['offset'] = [sum(x) for x in 
                               zip(metadata_dict['time'], metadata_dict['duration'])]

    # Depending on namespace, time might be in quarter notes instead of seconds.
    if ticks_to_sec_change:
        all_tempos = jam_file.annotations['tempo'][0].data
        tempo_onsets = [entry.time for entry in all_tempos]
        tempos = [entry.value for entry in all_tempos]
        metadata_dict['time'] = ticks_to_sec(metadata_dict['time'],
                                                   tempos, tempo_onsets)
        metadata_dict['offset'] = ticks_to_sec(metadata_dict['offset'],
                                                    tempos, tempo_onsets)

    metadata_dict = clean_metadict(metadata_dict, keys_to_remove=keys_to_remove,
                                   reverse_string=reverse_string)
    pitch_key = 'value'
    if pitch_key not in metadata_dict:
        metadata_dict = pitch_from_fret(metadata_dict, pitch_key='value',
                                        string_key='string', fret_key='fret')
    metadata_dict = standardize_keys_dict(metadata_dict,  pitch_key=pitch_key,
                                          string_key='string', onset_key='time', 
                                          offset_key='offset',)
    return metadata_dict

def load_xml_as_dict(path, root_ind=1, pitch_key='pitch',
                    string_key='stringNumber', onset_key='onsetSec', 
                    offset_key='offsetSec', standardize_string_format=True):
    '''
    XML files differ so much that this cannot possibly account for all annotations in XML format.
    IT does however account for all IDMT annotations. Some efforts have been made for generalization,
    but there just is any guarantee.
    '''
    tree  = ET.parse(path)
    transcription_root = tree.getroot()[root_ind]
    key_list = []
    metadata_list = []
    for event in transcription_root:
        tag = [annotation.tag for annotation in event]
        metadata = [annotation.text for annotation in event]
        key_list.extend(tag)
        metadata_list.extend(metadata)
    metadata_dict = dict_from_lists(key_list, metadata_list)
    metadata_dict = standardize_keys_dict(metadata_dict,  pitch_key=pitch_key,
                                          string_key=string_key, onset_key=onset_key, 
                                          offset_key=offset_key)
    metadata_dict = clean_metadict(metadata_dict, 
                                   standardize_string_format=standardize_string_format)
    if 'fretNumber' in metadata_dict:
        metadata_dict['fret'] = metadata_dict['fretNumber']
    return metadata_dict

def offset_time_per_event(metadata_dict, 
                      onset_key='onset_time', string_key='string',
                      offset_or_onset_key='type', offset_key_val='note_off'):
    '''
    In MIDI files, the offset time for a note event is recorded as a separate event.
    In this function, a dictionary with separate values for offset events is converted 
    to a dictionary with the offset time as a key value instead.

    onset_key : str, but not delta time but actual time.
    '''
    metadata_dict['offset_time'] = metadata_dict['onset_time'].copy()
    # for each string, we get their times.
    for string in range(0, 6):
        times = []
        for idx in range(len(metadata_dict[onset_key])):
            if metadata_dict[string_key][idx] == string:
                times.append((idx, metadata_dict[onset_key][idx]))
        # the offset is the following time value. no need to check type.
        for idx in range(len(times) - 1):
            metadata_dict['offset_time'][times[idx][0]] = times[idx + 1][1]
    # remove offset note events, to be in line with format later.
    while True:
        try:
            idx = metadata_dict[offset_or_onset_key].index(offset_key_val)
            for key in metadata_dict.keys():
                metadata_dict[key].pop(idx)
        except ValueError:
            break
    return metadata_dict

def load_midi_as_dict(path, reverse_string=True):
    midi = mido.MidiFile(path)
    time = 0
    metadata_list = []
    key_list = []
    for message in midi:
        time += message.time
        if 'note' in message.type:
            message_attr = message.dict()
            if message.type == 'note_on' and message.velocity <= 0:
                message_attr['type'] = 'note_off'  
            attr_val = list(message_attr.values())
            attr_keys = list(message_attr.keys())
            metadata_list.extend(attr_val)
            metadata_list.append(time)
            key_list.extend(attr_keys)
            key_list.append('onset_time')
    metadata_dict = dict_from_lists(key_list, metadata_list)
    metadata_dict = clean_metadict(metadata_dict, reverse_string=reverse_string, 
                                    string_key='channel')
    metadata_dict = offset_time_per_event(metadata_dict, string_key='channel')
    metadata_dict = standardize_keys_dict(metadata_dict, pitch_key='note',
                                          string_key='channel', onset_key='onset_time', 
                                          offset_key='offset_time')
    return metadata_dict

def fret_from_pitch(metadata_dict, pitch_key='pitch', string_key='string', 
                    fret_key='fret'):
    '''Every metadata_dict has the pitch value if done correctly. From the pitch and string,
    the fret value is determined and added to the metadata_dict. This is not necessary for 
    annotations from IDMT, but applying this function also on them is harmless.'''
    metadata_with_fret = metadata_dict.copy()
    metadata_with_fret[fret_key] = metadata_with_fret[pitch_key].copy()
    for idx, pitch in enumerate(metadata_with_fret[pitch_key]):
        string = metadata_with_fret[string_key][idx]
        midi_pitch = int(round(pitch)) 
        fret_value = midi_pitch - MIDI_NOTE_NUMBERS[string]
        metadata_with_fret[fret_key][idx] = fret_value
    return metadata_with_fret

def pitch_from_fret(metadata_dict, pitch_key='pitch', string_key='string', 
                    fret_key='fret'):
    metadata_with_fret = metadata_dict.copy()
    metadata_with_fret[pitch_key] = metadata_with_fret[fret_key].copy()
    for idx, fret in enumerate(metadata_with_fret[fret_key]):
        string = metadata_with_fret[string_key][idx]
        pitch_value = fret + MIDI_NOTE_NUMBERS[string]
        metadata_with_fret[pitch_key][idx] = pitch_value
    return metadata_with_fret

def file_grouper(dataset, path_format=True, return_match_dict=False):
    '''
    This function will associate which annotations and audio files correspond to each other.
    It will output two lists, which are the paths / file names 
    of the annotations and audio files respectively.
    It determines which annotations and audio files correspond to each other using an 
    affix structure given by a regular expression. 
    
    dataset : string which is the name of the dataset and not the path.
    '''
    _, annot_path, audio_path = get_audio_annot_folder_path(dataset)
    annot_files, audio_files = os.listdir(annot_path), os.listdir(audio_path)
    match_dict = {}
    for ann_file in annot_files:
        if path_format:
            ann_name = annot_path + '/' + ann_file
        else:
            ann_name = ann_file
        ann_affix = re.search(DATASET_AFFIX_DICT[dataset], ann_file).group()
        if ann_affix in match_dict:
            match_dict[ann_affix][0].append(ann_name)
        else:
            match_dict[ann_affix] = ([ann_name],[])
    for aud_file in audio_files:
        if path_format:
            aud_name = audio_path + '/' + aud_file
        else:
            aud_name = aud_file
        aud_affix = re.search(DATASET_AFFIX_DICT[dataset], aud_file).group()
        if aud_affix in match_dict:
            match_dict[aud_affix][1].append(aud_name)
        else:
            match_dict[aud_affix] = ([],[aud_name])
    if return_match_dict:
        return match_dict
    else:
        grouped_files = list(match_dict.values())
        return grouped_files

def dict_to_label_format(metadata_dict, times, onset_key='onset_time', 
                         offset_key='offset_time', string_key='string', fret_key='fret'):
    '''
    metadata_dict : dictionary
    onset_key : string which is the current key of metadata_dict 
                which relates to the onset of the note.
    offset_key : string which is the current key of metadata_dict 
                which relates to the onset of the note.
    times : list that states the time values which the output dictionary will occupy. 

    returns : torch.tensor of shape (6, len(times)). The values in the tensor will all 
                be from -1 to 22.
    '''
    times = torch.tensor(times)
    labels = -1 * torch.ones(6, len(times)) 
    for idx, string in enumerate(metadata_dict[string_key]):
        string = int(string)
        onset_time = metadata_dict[onset_key][idx]
        offset_time = metadata_dict[offset_key][idx]
        fret_value = int(metadata_dict[fret_key][idx])
        within_interval = torch.logical_and((times >= onset_time),
                                            (times <= offset_time))
        labels[string][within_interval] = fret_value
    return labels

def index_metadata_dict(metadata_dict, note_idx, pitch_key='pitch'):
    onset_time = metadata_dict['onset_time'][note_idx]
    offset_time = metadata_dict['offset_time'][note_idx]
    string = metadata_dict['string'][note_idx]
    pitch = metadata_dict[pitch_key][note_idx]
    return onset_time, offset_time, string, pitch

def save_dict_as_jams(path, metadata_dict):
    '''
    metadata_dict :
    '''
    metadata_out = jams.FileMetadata(title=path,
                      duration=23)
    jams_out = jams.JAMS(file_metadata=metadata_out)
    ann_out = []
    # create annotation for each string
    for string_idx in range(6):
        ann_meta_out = jams.AnnotationMetadata(data_source=str(string_idx))
        ann = jams.Annotation(namespace="note_midi",annotation_metadata=ann_meta_out)
        ann_out.append(ann)
    # annotate
    for note_idx in range(len(metadata_dict['onset_time'])):
        on_t, of_t, string, pitch = index_metadata_dict(metadata_dict, note_idx)
        ann_out[string].append(time=on_t, duration=of_t - on_t, value=pitch, confidence=0)
    # save annotations to JAMS object
    for string_idx in range(6):
        jams_out.annotations.append(ann_out[string_idx])
    jams_out.save(path)

def load_gp5_as_dict(path, qn_offset=1):
    '''
    GOAT has note events misaligned by 1 quarter note, so this takes that into account.
    qn_offset : int (quarter-notes by how much to offset onset and offset times)
    '''
    gp_file = guitarpro.parse(path)
    metadata_dict = {'onset_time':[], 'offset_time':[], 'string':[], 'fret':[]}
    tempos = [gp_file.tempo]
    tempo_onsets = [0]
    for track in gp_file.tracks:
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    time = beat.start
                    beat_dur = beat.duration.time
                    if beat.effect.mixTableChange is not None:
                        if beat.effect.mixTableChange.tempo is not None:
                            tempos.append(beat.effect.mixTableChange.tempo.value)
                            tempo_onsets.append(time)
                    for note in beat.notes:
                        string = note.string
                        fret = note.value
                        offset = time + beat_dur * note.durationPercent

                        # GOAT annotations are 1 quarternote behind CQT.
                        buffer = QUARTER_TIME * qn_offset
                        time = time - buffer if time >= buffer else 0
                        offset = offset - buffer if offset >= buffer else 0 
                        metadata_dict['onset_time'].append(time)
                        metadata_dict['offset_time'].append(offset)
                        metadata_dict['string'].append(string)
                        metadata_dict['fret'].append(fret)
    metadata_dict['onset_time'] = ticks_to_sec(metadata_dict['onset_time'],
                                            tempos, tempo_onsets)
    metadata_dict['offset_time'] = ticks_to_sec(metadata_dict['offset_time'],
                                            tempos, tempo_onsets)
    metadata_dict = clean_metadict(metadata_dict, standardize_string_format=True, 
                    reverse_string=True, string_key='string', keys_to_remove=None)
    return metadata_dict

annot_loader_dict = {'.jams': load_jam_as_dict,
                    '.xml': load_xml_as_dict,
                    '.midi': load_midi_as_dict,
                    '.mid': load_midi_as_dict,
                    '.gp5': load_gp5_as_dict,}