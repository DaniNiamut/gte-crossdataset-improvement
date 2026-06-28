from data_handling.load import *
from utils.utils import error_handling_makedirs
from data_handling.picker_functions import dataset_picker_dict

class DataPreprocessor():
    '''
    Preprocesses audio and annotations into Spectograms and their frame-level labels.
    '''
    def __init__(self, sample_rate=22050, hop_length=512,
                 cqt_n_bins=192, cqt_bins_per_octave=24,
                 trim=False, buffer=0.5, cut_off=10):
        '''
        we don't have it, but TabCNN had [c, m, cm, s] -> cqt, melspec, cm, s
        '''
        self.sr = sample_rate
        self.hop_length = hop_length
        self.cqt_n_bins = cqt_n_bins
        self.cqt_bins_per_octave = cqt_bins_per_octave
        self.h5_name = 'datasets.hdf5'
        self.trim_count = 0
        self.trim = trim
        self.buffer = buffer
        self.cut_off = cut_off

        # If a new dataset were to be added, it would require additions to 
        # DATASET_AFFIX_DICT. But if this new dataset contains a new annotation format,
        #  it would have to be added to dataset_loader_dict. 
        # One possible drawback of this method is that one might want multiple loading 
        # functions per file extension. This will be treated as beyond the 
        # scope of this project though.
        self.annot_loader_dict = annot_loader_dict
        
        # For creation of metadata.csv.
        self.id_nr = 0
        self.origins = []
        self.audio_ids = []
        self.start_idces = []
        self.end_idces = []
        self.num_note_events = []

    def save_h5(self, dataset, key, data, labels, idle_stats):
        h5_path = SPEC_AND_LABEL_PATH + '/' + self.h5_name
        if not os.path.isdir(SPEC_AND_LABEL_PATH):
            error_handling_makedirs(SPEC_AND_LABEL_PATH)

        f = h5py.File(h5_path, 'a')
        
        if f'{dataset}/labels' not in f:
            f.create_dataset(name=f'{dataset}/labels', data=labels,
                             maxshape=(6, None), chunks=(6, 128))
            f.create_dataset(name=f'{dataset}/specs', data=data,
                             maxshape=(self.cqt_n_bins, None), chunks=(192, 128),)
            f.create_dataset(name=f'{dataset}/idle_stats', data=idle_stats,
                             maxshape=(6, None), chunks=(6, 128),)
            new_ds_len = data.shape[1]
            prev_ds_len = 0
        else:
            ds_spec_shape = list(f[f'{dataset}/specs'].shape)
            ds_labels_shape = list(f[f'{dataset}/labels'].shape)
            prev_ds_len = ds_spec_shape[1]
            data_len = labels.shape[1]
            new_ds_len = prev_ds_len + data_len
            ds_spec_shape[1] = new_ds_len
            ds_labels_shape[1] = new_ds_len
            f[f'{dataset}/specs'].resize(tuple(ds_spec_shape))
            f[f'{dataset}/labels'].resize(tuple(ds_labels_shape))
            f[f'{dataset}/idle_stats'].resize(tuple(ds_labels_shape))
            f[f'{dataset}/specs'][:, prev_ds_len:new_ds_len] = data
            f[f'{dataset}/labels'][:, prev_ds_len:new_ds_len] = labels
            f[f'{dataset}/idle_stats'][:, prev_ds_len:new_ds_len] = idle_stats
        self.id_nr += 1
        audio_id = key + '_' + str(self.id_nr)
        self.audio_ids.append(audio_id)
        self.start_idces.append(prev_ds_len)
        self.end_idces.append(new_ds_len)
        self.origins.append(dataset)
        f.flush()
        f.close()

    def main(self, dataset,):
        '''
        Given a dataset it applies file_grouper. Then it chooses which audio and 
        annot files to use if there are multiple using the provided audio_picker 
        or annot_picker function.

        Two example use cases: 
        1. EGDB is a dataset of multiple audio files per same annotation, but 
            we want to use all audios as separate samples.
        2. GuitarTECHS contain two audio files per annotation, yet we only want to
            use one of them.
        
        audio_picker must be a function which takes a list of strings and outputs 
        a list of strings back. annot_picker must be a function which takes a list
        of strings and outputs only one of those strings back. Each element from the list
        outputted from audio_picker will be treated as it's own sample, with the 
        corresponding label being the string outputted from annot_picker. audio_picker and
        annot_picker are None by default. If None, the function will attempt to see if the class
        '''
        match_dict = file_grouper(dataset=dataset, path_format=True, return_match_dict=True)
        # function that returns a string
        annot_picker = dataset_picker_dict[dataset][0]
        # function that returns a list with string(s).
        audio_picker = dataset_picker_dict[dataset][1]
        # kwargs to annot_picker
        annot_kwargs = dataset_picker_dict[dataset][2]
        # kwargs to audio_picker
        audio_kwargs = dataset_picker_dict[dataset][3]

        for key in tqdm(match_dict.keys(), mininterval=1800):
            possible_annot = match_dict[key][0]
            possible_audio = match_dict[key][1]
            if (len(possible_audio) == 0) or (len(possible_annot) == 0):
                continue
            annot = annot_picker(possible_annot, **annot_kwargs)
            possible_audio = audio_picker(possible_audio, **audio_kwargs)

            for audio in possible_audio:
                data, labels, idle_stats = self.generate_spec_and_labels(audio, annot)
                self.save_h5(dataset, key, data, labels, idle_stats)

    def generate_metadata(self, name='metadata'):
        path = SPEC_AND_LABEL_PATH + '/' + name + '.csv'

        key_csv = ['dataset_origin', 'audio_id', 'start_idx', 'end_idx', 'num_note_events']
        val_csv = [self.origins, self.audio_ids, self.start_idces, self.end_idces,
                   self.num_note_events]
        # Source - https://stackoverflow.com/a/23613603
        # Posted by Tim Pietzcker, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-03-06, License - CC BY-SA 3.0
        with open(path, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(key_csv)
            writer.writerows(zip(*val_csv))

    def generate_spec(self, audio_path):
        # In TabCNN repo, some preprocessing happens. 
        # These are done by default with librosa.load.
        wav, _ = librosa.load(audio_path, sr=self.sr)
        wav = librosa.util.normalize(wav)
        data = librosa.cqt(wav,
                hop_length=self.hop_length, 
                sr=self.sr, 
                n_bins=self.cqt_n_bins, 
                bins_per_octave=self.cqt_bins_per_octave)
        data = torch.abs(torch.from_numpy(data))
        self.frame_indices = range(data.shape[1])
        return data

    def generate_labels(self, annot_path, save_num_note_events=False):
        frame_indices = self.frame_indices
        # times[i] = frame_indices[i] * hop_length / sr.
        # duration = len(times) * hop_length / sr.
        # fps = sr / hop_length
        times = librosa.frames_to_time(frame_indices, 
                                        sr=self.sr, 
                                       hop_length=self.hop_length)
        _, ext = os.path.splitext(annot_path)
        metadata_dict = self.annot_loader_dict[ext](annot_path)
        if 'fret' not in metadata_dict:
            metadata_dict = fret_from_pitch(metadata_dict)
        labels = dict_to_label_format(metadata_dict, times)
        if save_num_note_events is True:
            self.num_note_events.append(len(metadata_dict['fret']))
        return labels
    
    def calc_idle_stats(self, labels):
        secs_since_last_fretted = torch.zeros_like(labels)
        for string in range(6):
            count = 0
            for sample in range(labels.shape[1]):
                if labels[string, sample] == -1:
                    count += 1
                    secs_since_last_fretted[string, sample] = count * self.hop_length / self.sr
                else:
                    count = 0
        return secs_since_last_fretted
    
    def generate_spec_and_labels(self, audio_path, annot_path):
        data = self.generate_spec(audio_path)
        labels = self.generate_labels(annot_path, save_num_note_events=True)
        idle_stats = self.calc_idle_stats(labels)
        if self.trim is True:
            old_labels = labels
            data, labels, idle_stats = self.trim_excess_silences(data, labels, idle_stats)
            if old_labels.shape[1] != labels.shape[1]:
                self.trim_count += 1
        return data, labels, idle_stats
    
    def trim_excess_silences(self, data, labels, idle_stats):
        '''
        Abridges the labels and CQT when there is an extended amount of silence.
        This often occurs in SynthTab and PLUS, as there are often long stretches
        of silence in the recordings.

        cut_off : float (in seconds)
        buffer : float (in seconds)
        '''
        is_silent = torch.all(labels == -1, dim=0)
        if self.buffer is None and self.cut_off is None:
            return data[:, ~is_silent], labels[:, ~is_silent], idle_stats[:, ~is_silent]
        fps = self.sr / self.hop_length
        frames_buffer = round(self.buffer * fps)
        frames_cut_off = round(self.cut_off * fps)
        silent_int = is_silent.int()
        diff = torch.diff(silent_int, dim=0, prepend=torch.tensor([0],))
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0]

        if is_silent[-1]:
            ends = torch.cat([ends, torch.tensor([len(is_silent)],)])
        lengths = ends - starts
        long_silence_mask = lengths >= frames_cut_off
        trim_starts = starts[long_silence_mask] + frames_buffer
        trim_ends = ends[long_silence_mask]
        to_remove_delta = torch.zeros(len(is_silent) + 1,)
        valid_trim = trim_starts < trim_ends
        ts = trim_starts[valid_trim]
        te = trim_ends[valid_trim]
        to_remove_delta[ts] = 1
        to_remove_delta[te] = -1
        to_remove = torch.cumsum(to_remove_delta, dim=0)[:-1].bool()

        return data[:, ~to_remove], labels[:, ~to_remove], idle_stats[:, ~to_remove]