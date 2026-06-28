import torch
import pickle
import random
import pandas as pd
import socket
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.misc import *
from utils.utils import identity_function
from models.tabcnn import TabCNN
from data_handling.dataset import DatasetsGTE
from models.metrics import *
from torch.utils.data import ConcatDataset
from utils.utils import error_handling_makedirs
'''
This file contains classes that retrieve the experiment parameters.
'''

# Base

class Experiment():
    def __init__(self):

        # Dataset & dataloader params
        # this is a list of which datasets to make dataloaders out of
        self.datasets_to_incl = ['GuitarSet',]
        # this is a list of which dataloaders not to use in training. 
        # this means the dataloaders are created if included in self.datasets_to_incl
        # and used only for combining datasets if toggled on and test set evaluation.
        self.skip_datasets = []
        self.partition_by_tone = False
        self.combine_datasets = False
        self.test_only_datasets = None
        self.audio_ids_to_excl = None
        self.hop_length = 9 // 2

        # Model params
        self.norm_fn = None
        self.input_state_dict = None
        self.frame_length = 9
        self.num_frets = 22
        self.cqt_bins = 192

        # Loss fn params
        self.distance_weight_fn = None
        self.class_weights = None
        self.gamma = 0
        self.sample_weight_fn = None

        # Train params
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 1000
        self.batch_size = 64
        self.lr = 1.0
        self.num_workers = 4
        self.optimizer_func = torch.optim.Adadelta
        self.lr_scheduler = ReduceLROnPlateau
        self.patience = 10
        self.lr_factor = 0.1
        self.do_early_stopping = True
        
        # misc constants
        self.prev_lr = self.lr
        self.best_loss = 1e100
        self.best_lr_run_val_loss = 1e100
        self.cv_folds = None
        self.current_fold = 0

        self.save_path = PREV_MODELS_PATH + '/Experiment/'
        metadata_path = SPEC_AND_LABEL_PATH + '/metadata.csv'
        self.metadata_df = pd.read_csv(metadata_path)

        self.label_metrics = {'tab' : [tab_excl_silence, True, p_r_f_metrics],
                         'pitch' : [tab2pitch, True, p_r_f_metrics],
                         'sonority' : [tab2son, False, identity_function]}
        self.label_metric_names = {'tab' : ['tab_precision', 'tab_recall', 'tab_f1'],
                            'pitch' : ['pitch_precision', 'pitch_recall', 'pitch_f1'],
                            'sonority' : ['ts', 'tsfss', 'tsfms',
                                          'tssfs', 'tss', 'tssfms',
                                          'tmsfs', 'tmsfss', 'tms']}
        self.additional_metrics = [tab_disamb_rate]
        self.additional_metric_names =  ['tdr']
        self.log_info_names = ['ds_eval_on', 'ds_train_on', 'loader', 'learning_rate']
        self.metrics_dict = None
        # metrics_dict additional key
        self.print_fn = print
        self.print_fn(f'Training on {self.device}\n',
                f'Torch version: {torch.__version__}\n',
                f'Cuda Available: {torch.cuda.is_available()}\n')
        
    def stopping_criterion(self, val_loss, curr_lr, model):
        if val_loss < self.best_lr_run_val_loss:
                self.best_lr_run_val_loss = float(val_loss)
                self.curr_best_model = model.state_dict()
        if curr_lr != self.prev_lr:
            if self.best_loss <= self.best_lr_run_val_loss:
                return True
            else:
                self.best_loss = float(self.best_lr_run_val_loss)
                self.best_lr_run_val_loss = 1e100
                self.prev_lr = curr_lr
                model.load_state_dict(self.curr_best_model)

    def partition_guitarset_by_tone(self, dataset_df):
        gs_ids = dataset_df[dataset_df.dataset_origin == "GuitarSet"]["audio_id"]

        id_player_splits = (gs_ids.groupby(gs_ids.str.rsplit("_", n=0).str[0]))
        id_player_splits = id_player_splits.apply(list).to_list()
        num_players = len(id_player_splits) 

        test_idx = self.current_fold % num_players
        val_idx = (self.current_fold + 1) % num_players
        test_split = id_player_splits[test_idx]
        val_split = id_player_splits[val_idx]
    
        train_split = []
        for idx in range(num_players):
            if idx != test_idx and idx != val_idx: 
                train_split.extend(id_player_splits[idx])
        return [train_split, val_split, test_split]

    def partition_audio_ids_to_excl(self, dataset_df : pd.DataFrame, splits,):
        if self.audio_ids_to_excl is not None:
            dataset_df = dataset_df[~dataset_df.audio_id.isin(self.audio_ids_to_excl)]
        
        dataset_df = dataset_df.reset_index(drop=True)
        audio_id_to_perf = lambda x: x.rsplit('_', 1)[0] if '_' in x else x
        dataset_df['perf_root'] = dataset_df['audio_id'].apply(audio_id_to_perf)
        perf_groups = dataset_df.groupby('perf_root')['num_note_events'].sum().reset_index()
        sum_frames = perf_groups['num_note_events'].sum()
        perf_to_audio_ids = dataset_df.groupby('perf_root')['audio_id'].apply(list).to_dict()
        perf_roots_list = perf_groups['perf_root'].tolist()
        random.shuffle(perf_roots_list)
        
        output_partitions = tuple()
        i = 0
        
        for split in splits:
            chosen_audio_ids = []
            sum_entries = 0
            
            while (sum_entries < split * sum_frames) and (i < len(perf_roots_list)):
                current_perf = perf_roots_list[i]
                matching_ids = perf_to_audio_ids[current_perf]
                chosen_audio_ids.extend(matching_ids)
                subsec = perf_groups['perf_root'] == current_perf
                sum_entries += perf_groups.loc[subsec, 'num_note_events'].values[0]
                i += 1
            output_partitions += chosen_audio_ids[:-1],
            i -= 1
        return output_partitions

    def create_metrics_dict(self):
        metric_names = []
        for _, names in self.label_metric_names.items():
            metric_names += names
        keys = ['loss'] + metric_names + self.additional_metric_names + self.log_info_names
        metrics_dict = {key : [] for key in keys}
        return metrics_dict
    
    def get_dataloaders(self, datasets_to_incl, hop_length, splits=[0.7, 0.1, 0.2],
                        combine_datasets=False):
        '''
        This functions returns dataloaders for the amount of splits
        for a defined DatasetsGTE object. 
        The function returns a dictionary with the elements of datasets_to_incl as keys
        and their corresponding dataloader as values.
        '''
        dataloaders = None
        if datasets_to_incl is None:
            return None
        for dataset in datasets_to_incl:
            dataset_df = self.metadata_df[self.metadata_df.dataset_origin==dataset]
            dataloaders = self.create_dataloader(dataset, [dataset], splits, dataset_df,
                                                hop_length, dataloaders)
        if combine_datasets is True:
            dataset_df = self.metadata_df
            training_sets = [dataloaders[dataset][0].dataset for dataset in dataloaders]
            val_sets = [dataloaders[dataset][1].dataset for dataset in dataloaders]
            training_sets = ConcatDataset(training_sets)
            val_sets = ConcatDataset(val_sets)
            training_loader = DataLoader(dataset=training_sets, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
            val_loader = DataLoader(dataset=val_sets, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
            dataloaders['Hodgepodge'] = (training_loader, val_loader, None)
        return dataloaders
    
    def create_dataloader(self, dataset_name, datasets_to_incl, splits, dataset_df,
                           hop_length, dataloaders=None):
        n_splits = len(splits)
        use_idle_stats = False if self.sample_weight_fn is None else True
        dataloaders = dict() if dataloaders is None else dataloaders
        if self.partition_by_tone is True and dataset_name == 'GuitarSet':
            if n_splits != 3:
                raise TypeError('Should be 3 splits when partitioning GuitarSet by tone.')
            id_splits = self.partition_guitarset_by_tone(dataset_df)
        else:
            id_splits = self.partition_audio_ids_to_excl(dataset_df, splits)

        output_loaders = tuple()
        for i in range(n_splits):
            shuffle = False if i > 0 or n_splits == 1 else True
            # make ids_to_excl, example: train_ids_excl = val_ids + test_ids
            ids_to_excl = id_splits[(i + 1) % n_splits] + id_splits[(i + 2) % n_splits]
            ids_to_excl = ids_to_excl if n_splits != 1 else []
            if self.audio_ids_to_excl is not None:
                ids_to_excl += self.audio_ids_to_excl
            partition = DatasetsGTE(self.num_frets, self.frame_length, hop_length,
                                    datasets_to_incl, audio_ids_to_excl=ids_to_excl,
                                    use_idle_stats=use_idle_stats)
            loader = DataLoader(dataset=partition, batch_size=self.batch_size,
                                shuffle=shuffle, num_workers=self.num_workers)
            output_loaders += loader,
        dataloaders[dataset_name] = output_loaders
        return dataloaders

    def collect_global_metrics(self, metrics_dict, log_info, loss, label_cms=None,):
        metrics_dict['loss'].append(loss)
        for i, log in enumerate(self.log_info_names):
            metrics_dict[log].append(log_info[i])
        if label_cms is not None:
            for label, loop_cm in label_cms.items():
                dict_keys = self.label_metric_names[label]
                global_metrics_fn = self.label_metrics[label][2]
                global_metrics = global_metrics_fn(loop_cm)
                for i, key in enumerate(dict_keys):
                    metrics_dict[key].append(global_metrics[i].item())
            recent_metrics = {key : metrics_dict[key][-1]
                              for key in metrics_dict if metrics_dict[key] != []}
            for i, metric_fn in enumerate(self.additional_metrics):
                metrics_dict[self.additional_metric_names[i]].append(metric_fn(**recent_metrics))
        else:
            for key in metrics_dict:
                if key in ['loss'] + self.log_info_names:
                    continue
                metrics_dict[key].append(None)
        return metrics_dict

    def training_run(self, dataset,):
        trainloader, validationloader, _ = self.dataloaders[dataset]
        if self.class_weights is True:
            freq = trainloader.dataset.class_freqs(device=self.device)
            ds_len = len(trainloader.dataset)
            class_weights = ds_len / ((self.num_frets + 2) * (freq + 1))
        else:
            class_weights = self.class_weights 
        model = TabCNN(dim_in=self.cqt_bins, num_frets=self.num_frets,
                       device=self.device, frame_width=self.frame_length,
                       norm_fn=self.norm_fn, sample_weight_fn=self.sample_weight_fn,
                       class_weights=class_weights, gamma=self.gamma,
                       distance_weight_fn=self.distance_weight_fn)
        if self.input_state_dict is not None:
            model.load_state_dict(self.input_state_dict)
        model = model.to(model.device)
        if self.metrics_dict is None:
            self.metrics_dict = self.create_metrics_dict()
        optimizer = self.optimizer_func(model.parameters(), lr=self.lr)
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer, mode='min', factor=self.lr_factor,
                                          patience=self.patience)
        else:
            scheduler = None
        
        for epoch in range(self.num_epochs): 
            train_loss = model.fit(dataloader=trainloader, optimizer=optimizer,)
            self.metrics_dict = self.collect_global_metrics(self.metrics_dict, 
                                                        [dataset, dataset, 'train', self.prev_lr],
                                                        train_loss)
            val_loss, val_cms = model.evaluate(validationloader, self.label_metrics)
            self.metrics_dict = self.collect_global_metrics(self.metrics_dict,
                                                        [dataset, dataset, 'val', self.prev_lr],
                                                        val_loss, val_cms)
            if scheduler is not None:
                scheduler.step(val_loss)
            criterion_output = self.stopping_criterion(val_loss=val_loss,
                                curr_lr=optimizer.param_groups[0]['lr'], model=model)
            if criterion_output is True and self.do_early_stopping is True:
                break
            if ((epoch) % 5 == 0) or criterion_output is True:
                self.print_fn((f'epoch {round(epoch, 3)}, ' +
                    f'lr-run loss : {round(self.best_lr_run_val_loss, 3)}, ' +
                    f'loss to beat : {round(self.best_loss, 3)}, ' +
                    f'lr : {round(self.prev_lr, 5)}'))
        
        model.load_state_dict(self.curr_best_model)
        # save test metrics.
        self.metrics_dict = self.collect_test_metrics(model=model, dataset=dataset,
                                                dataloaders=self.dataloaders,
                                                metrics_dict=self.metrics_dict,)
        if self.test_only_loaders is not None:
            self.metrics_dict = self.collect_test_metrics(model=model, dataset=dataset,
                                                dataloaders=self.test_only_loaders,
                                                metrics_dict=self.metrics_dict,)
        return model.state_dict()
    
    def collect_test_metrics(self, model : TabCNN, dataset,
                                    dataloaders, metrics_dict):
        new_log_info = [None, dataset, 'test', None]
        for test_dataset in dataloaders:
            new_log_info[0] = test_dataset
            # Choose idx=-1 to make sure it is the test set in dataloaders.
            dataloader = dataloaders[test_dataset][-1]
            # if the last item in the tuple is None then there is no test set.
            if dataloader is None:
                continue
            test_loss, test_metrics = model.evaluate(dataloader, self.label_metrics)
            metrics_dict = self.collect_global_metrics(metrics_dict, new_log_info,
                                                       test_loss, label_cms=test_metrics)
        return metrics_dict

    def run_experiment(self, save_results=True,
                       output_model_name='model_state',):
        error_handling_makedirs(self.save_path)
        self.dataloaders = self.get_dataloaders(datasets_to_incl=self.datasets_to_incl,
                                                hop_length=self.hop_length,
                                                combine_datasets=self.combine_datasets)
        self.test_only_loaders = self.get_dataloaders(datasets_to_incl=self.test_only_datasets,
                                                    hop_length=1, splits=[1],
                                                    combine_datasets=False)

        for dataset in self.dataloaders:
            if dataset in self.skip_datasets:
                continue
            self.print_fn(f'Currently training on {dataset}:')
            self.best_loss = 1e100
            self.prev_lr = self.lr
            self.best_lr_run_val_loss = 1e100
            state_dict = self.training_run(dataset=dataset)

            # save metrics dict and model
            torch.save(state_dict, self.save_path + f'{output_model_name}_{dataset}.pth')
        if save_results is True:
            with open(self.save_path + f'model_results.pkl', 'wb') as f:
                pickle.dump(self.metrics_dict, f)

    def run_cv(self):
        for fold in range(self.cv_folds):
            save_results = False if fold != self.cv_folds - 1 else True
            self.run_experiment(save_results=save_results,
                                output_model_name=f'model_state_{fold}')
            self.current_fold += 1

class CrossDataset(Experiment):
    def __init__(self):
        super().__init__()
        metadata_path = SPEC_AND_LABEL_PATH + '/metadata.csv'
        df = pd.read_csv(metadata_path)
        goat_metadata_path = SPEC_AND_LABEL_PATH + '/GOAT_metadata.csv'
        goat_df = pd.read_csv(goat_metadata_path)
        bad_audios = goat_df[goat_df.alignment_f_measure_fine <= 0.90]['item']
        root_names = df['audio_id'].str.rsplit('_', n=1).str[0]
        filename_to_id_map = df['audio_id'].groupby(root_names).apply(list).to_dict()
        self.audio_ids_to_excl = []
        for name in bad_audios:
            if name not in filename_to_id_map:
                continue
            self.audio_ids_to_excl += filename_to_id_map[name]
        self.datasets_to_incl = ['EGDB', 'GuitarSet', 'IDMT', 'GuitarTECHS', 'GOAT']
        self.test_only_datasets = ['EGSet12']
        self.num_epochs = 100
        self.save_path = PREV_MODELS_PATH + '/CrossDataset/'

# Weighted Approaches

class SampleWeighted(CrossDataset):

    def __init__(self):
        super().__init__()
        def get_leniency_sample_weights(idle_stats):
            '''
            idle_stats : tensor (batch, num_labels)
            '''
            sample_weights = torch.ones_like(idle_stats, device=self.device)
            sample_weights[(0.5 >= idle_stats) & (idle_stats > 0)] = 0.25
            return sample_weights
        self.sample_weight_fn = get_leniency_sample_weights
        self.class_weights = None
        self.gamma = 0
        self.save_path = PREV_MODELS_PATH + '/SampleWeighted/'

class DistanceWeighted(CrossDataset):
    '''
    Weighted using distance based weights
    '''
    def __init__(self):
        super().__init__()
        
        def physical_fret_dist(pred, target):
            target_indices = target.argmax(dim=-1)
            pred_indices = pred.argmax(dim=-1)
            weights = 1 / (self.num_frets + 1) * torch.abs(target_indices - pred_indices) + 1
            false_neg_silence = (target_indices == 0) & ~(pred_indices == 0)
            false_pos_silence = (pred_indices == 0) & ~(target_indices == 0)
            weights[false_neg_silence] = 2
            weights[false_pos_silence] = 2
            return weights

        self.sample_weight_fn = None
        self.class_weights = None
        self.distance_weight_fn = physical_fret_dist
        self.gamma = 0
        self.save_path = PREV_MODELS_PATH + '/DistanceWeighted/'

class FocalLoss(CrossDataset):

    def __init__(self):
        super().__init__()
        self.sample_weight_fn = None
        self.class_weights = None
        self.gamma = 2
        self.save_path = PREV_MODELS_PATH + '/FocalLoss/'

class ClassWeighted(CrossDataset):

    def __init__(self):
        super().__init__()
        self.class_weights = torch.ones((6, self.num_frets + 2), device=self.device)
        self.class_weights[:, 0] = 0.25
        self.gamma = 0
        self.save_path = PREV_MODELS_PATH + '/ClassWeighted/'

class FLWithAllWeights(SampleWeighted):

    def __init__(self):
        super().__init__()

        def physical_fret_dist(pred, target):
            target_indices = target.argmax(dim=-1)
            pred_indices = pred.argmax(dim=-1)
            weights = 1 / self.num_frets * torch.abs(target_indices - pred_indices) + 1
            false_neg_silence = (target_indices == 0) & ~(pred_indices == 0)
            false_pos_silence = (pred_indices == 0) & ~(target_indices == 0)
            weights[false_neg_silence] = 2
            weights[false_pos_silence] = 2
            return weights
        
        self.gamma = 2
        self.class_weights = True
        self.distance_weight_fn = physical_fret_dist
        self.save_path = PREV_MODELS_PATH + '/FLWithAllWeights/'

# Synthetic dataset training

class PreTrain(Experiment):
    def __init__(self, version=15):
        super().__init__()
        self.datasets_to_incl = ['SynthTab', 'PLUS']
        self.test_only_datasets = ['EGSet12', 'IDMT', 'GuitarSet', 'EGDB', 'GuitarTECHS', 'GOAT']
        self.num_epochs = 100
        self.save_path = PREV_MODELS_PATH + f'/PreTrain{version}K/'
        df = pd.read_csv(SPEC_AND_LABEL_PATH + '/metadata.csv')
        self.audio_ids_to_excl = list(df[df.dataset_origin=='PLUS']['audio_id'][version * 1e3:])
        self.audio_ids_to_excl += list(df[df.dataset_origin=='SynthTab']['audio_id'][version * 1e3:])

class PreTrainHodgepodge(PreTrain):
    def __init__(self, version=15):
        super().__init__(version=version)
        self.skip_datasets = ['SynthTab', 'PLUS']
        self.combine_datasets = True
        self.save_path = PREV_MODELS_PATH + f'/PreTrainHodgepodge{version}K/'
        half_vers = int(version * 1e3 / 2)
        df = pd.read_csv(SPEC_AND_LABEL_PATH + '/metadata.csv')
        self.audio_ids_to_excl = list(df[df.dataset_origin=='PLUS']['audio_id'][half_vers:])
        self.audio_ids_to_excl += list(df[df.dataset_origin=='SynthTab']['audio_id'][half_vers:])

# Fine-tuning on pre-trained models

class PreTrainFineTune(CrossDataset):
    def __init__(self, init_model : str, version=15,):
        '''
        init_model must be either "PLUS" or "SynthTab".
        '''
        super().__init__()
        self.num_epochs = 5000
        params = torch.load(PREV_MODELS_PATH + f'/PreTrain{version}K/model_state_{init_model}.pth')
        self.batch_size = 8
        self.lr = 0.1
        self.input_state_dict = params
        self.save_path = PREV_MODELS_PATH + f'/CrossDSWith{init_model}{version}K/'

class PreTrainFineTuneClassWeighted(PreTrainFineTune):
    '''
    Weighted using class based rates.
    '''
    def __init__(self, init_model : str, version=15):
        super().__init__(init_model=init_model, version=version)
        self.class_weights = torch.ones((6, self.num_frets + 2), device=self.device)
        self.class_weights[:, 0] = 0.25
        self.save_path = PREV_MODELS_PATH + f'/ClassWeightedWith{init_model}{version}K/'

class PreTrainHodgepodgeFineTuneClassWeighted(CrossDataset):
    def __init__(self, version=15,):
        super().__init__()
        self.num_epochs = 5000
        self.class_weights = torch.ones((6, self.num_frets + 2), device=self.device)
        self.class_weights[:, 0] = 0.25
        params = torch.load(PREV_MODELS_PATH + f'/PreTrainHodgepodge{version}K/model_state_Hodgepodge.pth')
        self.batch_size = 8
        self.lr = 0.1
        self.input_state_dict = params
        self.save_path = PREV_MODELS_PATH + f'/CrossDSWithHodgepodge{version}K/'

class PreTrainHodgepodgeFineTune(CrossDataset):
    def __init__(self, version=15,):
        super().__init__()
        self.num_epochs = 5000
        params = torch.load(PREV_MODELS_PATH + f'/PreTrainHodgepodge{version}K/model_state_Hodgepodge.pth')
        self.batch_size = 8
        self.lr = 0.1
        self.input_state_dict = params
        self.save_path = PREV_MODELS_PATH + f'/CrossDSWithHodgepodge{version}K/'

# Miscellaneous

class PreTrainFineTuneGuitarSet(CrossDataset):
    def __init__(self, init_model : str, version=15,):
        '''
        init_model must be either "PLUS" or "SynthTab".
        '''
        super().__init__()
        self.num_epochs = 5000
        self.datasets_to_incl = ['IDMT', 'GuitarSet', 'EGDB', 'GuitarTECHS', 'GOAT']
        self.skip_datasets = ['IDMT', 'EGDB', 'GuitarTECHS', 'GOAT']
        self.test_only_datasets = ['EGSet12']
        params = torch.load(PREV_MODELS_PATH + f'/PreTrain{version}K/model_state_{init_model}.pth')
        self.batch_size = 8
        self.lr = 0.1
        self.input_state_dict = params
        self.save_path = PREV_MODELS_PATH + f'/GuitarSetFix{init_model}{version}K/'

class OnlyGOAT(Experiment):   
    def __init__(self, f_measure_threshold=0.75):
        super().__init__()
        metadata_path = SPEC_AND_LABEL_PATH + '/metadata.csv'
        df = pd.read_csv(metadata_path)
        goat_metadata_path = SPEC_AND_LABEL_PATH + '/GOAT_metadata.csv'
        goat_df = pd.read_csv(goat_metadata_path)
        bad_audios = goat_df[goat_df.alignment_f_measure_fine <= f_measure_threshold]['item']
        root_names = df['audio_id'].str.rsplit('_', n=1).str[0]
        filename_to_id_map = df['audio_id'].groupby(root_names).apply(list).to_dict()
        self.audio_ids_to_excl = []
        for name in bad_audios:
            if name not in filename_to_id_map:
                continue
            self.audio_ids_to_excl += filename_to_id_map[name]

        self.datasets_to_incl = ['GOAT']#, 'IDMT'] # just because it is small
        self.test_only_datasets = ['EGSet12',]# 'EGDB', 'GuitarSet', 'IDMT', 'GuitarTECHS']
        self.save_path = PREV_MODELS_PATH + '/GOAT/'

class Hodgepodge(CrossDataset):

    def __init__(self):
        super().__init__()
        self.datasets_to_incl = ['GuitarSet', 'IDMT', 'GuitarTECHS', 'GOAT']
        self.skip_datasets = ['GuitarSet', 'IDMT', 'GuitarTECHS', 'GOAT']
        self.combine_datasets = True
        self.partition_by_tone = True
        self.test_only_datasets = ['EGSet12', 'EGDB']
        self.save_path = PREV_MODELS_PATH + '/Hodgepodge/'

class MinMaxNorm(CrossDataset):

    def __init__(self):
        super().__init__()
        def log_min_max_norm(x):
            log_x = torch.log2(1 + 10 * x)
            return  log_x / (torch.amax(log_x, dim=(-2, -1), keepdim=True) + 1e-6)
        self.norm_fn = log_min_max_norm
        self.save_path = PREV_MODELS_PATH + '/MinMaxNorm/'

class PipelineTest(Experiment):
    '''
    Test as in, see if pipeline works.
    '''
    def __init__(self):
        super().__init__()
        self.datasets_to_incl = ['EGDB', 'IDMT'] # just because it is small
        self.test_only_datasets = ['EGSet12']
        self.combine_datasets = False
        df = pd.read_csv(SPEC_AND_LABEL_PATH + '/metadata.csv')
        self.audio_ids_to_excl = list(df[df.dataset_origin=='EGDB']['audio_id'][:-5])
        self.audio_ids_to_excl += list(df[df.dataset_origin=='IDMT']['audio_id'][:-5])
        self.audio_ids_to_excl += list(df[df.dataset_origin=='EGSet12']['audio_id'][:-5])
        self.num_epochs = 10
        self.patience = 2
        self.lr_factor = 0.01
        self.save_path = PREV_MODELS_PATH + '/PipelineTest/'