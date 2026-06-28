import h5py
import torch
from collections import Counter
from torch.utils.data import Dataset

from utils.misc import *
from utils.utils import load_csv_as_dict

class DatasetsGTE(Dataset):
    '''
    This class will load any dataset in the provided h5py file assuming it is in the expected format
    '''
    def __init__(self, num_frets, frame_length=9, hop_length=1, 
                datasets_to_incl=None, audio_ids_to_excl=None,
                use_idle_stats=False):
        '''
        num_frets : int
        frame_length : int
        hop_length : int s.t frame_overlap <= frame_length.
        cqt_bins : int (192)
        datasets : list of strings, where each string must be an entry in the dataset_origin column
        of metadata.csv. If None, all datasets will be included.
        audio_ids_to_excl : list of strings, where each string must be an entry in the audio_id
        idle_stats : bool, if True, __getitem__ retrieves the idle stat as well.
        column of metadata.csv. If None, no audio_ids will be excluded.
        '''
        self.datasets_path = SPEC_AND_LABEL_PATH + '/datasets.hdf5'
        metadata_path = SPEC_AND_LABEL_PATH + '/metadata.csv'
        self.feature_key, self.label_key, self.idle_stats_key = 'specs', 'labels', 'idle_stats'
        metadata_dict = load_csv_as_dict(metadata_path)

        if not isinstance(audio_ids_to_excl, list):
            audio_ids_to_excl = []
        if isinstance(datasets_to_incl, list):
            audio_ids = [metadata_dict['audio_id'][idx] for idx, dataset 
                        in enumerate(metadata_dict['dataset_origin']) 
                        if dataset in datasets_to_incl]
        else:
            audio_ids = metadata_dict['audio_id']
        audio_ids = [audio_id for audio_id in audio_ids if not audio_id in audio_ids_to_excl]

        self.audio_ids = audio_ids
        self.num_frets = num_frets
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.h5obj = None
        self.use_idle_stats = use_idle_stats

        temp_dataset_names = []
        temp_indices = []
        temp_pads = []

        half_win = self.frame_length // 2

        for i in range(len(metadata_dict['audio_id'])):
            audio_id = metadata_dict['audio_id'][i]
            dataset = metadata_dict['dataset_origin'][i]
            
            if datasets_to_incl and dataset not in datasets_to_incl:
                continue
            if audio_id in audio_ids_to_excl:
                continue

            start_idx = int(metadata_dict['start_idx'][i])
            end_idx = int(metadata_dict['end_idx'][i])
            
            h_indices = torch.arange(start_idx, end_idx, self.hop_length, dtype=torch.int32)
            num_frames = h_indices.shape[0]
            
            if num_frames == 0:
                continue

            dist_from_start = h_indices - start_idx
            dist_from_end = (end_idx - 1) - h_indices
            
            pad_left = torch.clamp(half_win - dist_from_start, min=0).to(torch.int8)
            pad_right = torch.clamp(half_win - dist_from_end, min=0).to(torch.int8)

            temp_dataset_names.append([dataset] * num_frames)
            temp_indices.append(h_indices)
            
            temp_pads.append(torch.stack([pad_left, pad_right], dim=1))

        self.map_datasets = [item for sublist in temp_dataset_names for item in sublist]
        
        self.map_hdf5_idx = torch.cat(temp_indices)
        self.map_pads = torch.cat(temp_pads, dim=0)

    def label_reshape(self, label):
        '''
        label for a frame as input. The output is the corresponding 
        one-hot encoded value.
        '''
        plot_labels = torch.zeros([6, self.num_frets + 2],
                                   dtype=torch.float32, device=label.device)

        one_hot_idces = (label + 1).long()
        one_hot_idces[one_hot_idces < 0] = 0
        one_hot_idces[one_hot_idces >= (self.num_frets + 2)] = 0
        string_idces = torch.arange(6, dtype=torch.int8, device=label.device).long()
        plot_labels[string_idces, one_hot_idces] = 1.0
        return plot_labels
    
    def cqt_reshape(self, hdf5_ds, idx, pad_l, pad_r):
        """
        Fetches a window of frame_length centered at idx.
        Pads with zeros if the window goes out of bounds.
        """
        half_win = self.frame_length // 2
        read_start = idx - half_win + pad_l
        read_end = idx + half_win + 1 - pad_r
        
        data = torch.from_numpy(hdf5_ds[:, read_start:read_end]).float()
        
        if pad_l > 0 or pad_r > 0:
            data = torch.nn.functional.pad(data, (pad_l, pad_r), value=0)

        return data.permute(1, 0).unsqueeze(0)

    def __getitem__(self, index):
        if self.h5obj is None:
            self.h5obj = h5py.File(self.datasets_path, mode='r')

        dataset_name = self.map_datasets[index]
        hdf5_idx = self.map_hdf5_idx[index].item()
        pad_l = self.map_pads[index, 0].item()
        pad_r = self.map_pads[index, 1].item()
        
        features_ds = self.h5obj[dataset_name][self.feature_key]
        label = self.h5obj[dataset_name][self.label_key][:, hdf5_idx]
        
        features = self.cqt_reshape(features_ds, hdf5_idx, pad_l, pad_r)
        
        label_raw = torch.from_numpy(label)
        label = self.label_reshape(label_raw)
        
        if self.use_idle_stats is True:
            idle_stats = self.h5obj[dataset_name][self.idle_stats_key][:, hdf5_idx]
            return features, label, idle_stats
        else:
            return features, label
    
    def __len__(self,):
        return self.map_hdf5_idx.shape[0]
    
    def class_freqs(self, device=None, chunk_size=50000):
        # [6, num_frets + 2]
        num_classes = self.num_frets + 2
        total_counts = torch.zeros([6, num_classes], dtype=torch.float32, device=device)
        
        if self.h5obj is None:
            self.h5obj = h5py.File(self.datasets_path, mode='r')

        dataset_counts = Counter(self.map_datasets)
        
        # Track unique datasets while preserving order
        unique_datasets = []
        for d in self.map_datasets:
            if not unique_datasets or unique_datasets[-1] != d:
                unique_datasets.append(d)

        global_idx = 0
        for dataset_name in unique_datasets:
            label_ds = self.h5obj[dataset_name][self.label_key]
            num_frames_in_ds = dataset_counts[dataset_name]
            
            sub_indices = self.map_hdf5_idx[global_idx : global_idx + num_frames_in_ds].numpy()
            
            for i in range(0, len(sub_indices), chunk_size):
                chunk_idx = sub_indices[i : i + chunk_size]
                chunk = torch.from_numpy(label_ds[:, chunk_idx])
                labels_chunk = chunk.to(device, dtype=torch.long) + 1
                
                for string_idx in range(6):
                    counts = torch.bincount(labels_chunk[string_idx], minlength=num_classes)
                    total_counts[string_idx] += counts.float()

            global_idx += num_frames_in_ds

        self.h5obj.close()
        self.h5obj = None
        return total_counts