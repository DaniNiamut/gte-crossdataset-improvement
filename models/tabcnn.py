# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>
# This is an altered version of the TabCNN offered in the amt-tools package.
import torch
from torch import nn
from models.metrics import multi_label_focal_loss, confusion_matrix
from utils.utils import identity_function, log_compression_norm
from tqdm import tqdm

class TabCNN(nn.Module):
    '''
    Implements the TabCNN model (http://archives.ismir.net/ismir2019/paper/000033.pdf).
    '''

    def __init__(self, dim_in, frame_width, in_channels=1,
                 device='cuda:0', model_complexity=4,
                 num_frets=22, patience=5, min_delta=0,
                 norm_fn=None, sample_weight_fn=None,
                 class_weights=None, gamma=0,
                 distance_weight_fn=None):
        '''
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        ...
        '''
        self.device = device
        self.dim_in = dim_in
        self.in_channels = in_channels
        self.model_complexity = model_complexity
        self.norm_fn = identity_function if norm_fn is None else norm_fn
        self.frame_width = frame_width
        self.min_delta = min_delta
        self.patience = patience
        self.num_strings = 6 
        self.num_classes = num_frets + 2
        self.class_weights = class_weights
        self.sample_weight_fn = sample_weight_fn
        self.distance_weight_fn = distance_weight_fn
        self.use_idle_stats = False if sample_weight_fn is None else True
        self.gamma = gamma
        # Total number of output neurons
        self.dim_out = self.num_strings * self.num_classes
        self.counter = 0

        nn.Module.__init__(self)

        # Number of filters for each stage
        nf1 = 32 * self.model_complexity
        nf2 = 64 * self.model_complexity
        nf3 = nf2

        # Kernel size for each stage
        ks = (3, 3)

        # Reduction size for each stage
        rd1 = (2, 2)

        # Dropout percentages for each stage
        dp1 = 0.25
        dp2 = 0.50

        self.conv = nn.Sequential(
            # 1st convolution
            nn.Conv2d(in_channels=self.in_channels, out_channels=nf1, kernel_size=ks),
            # Activation function
            nn.ReLU(),
            # 2nd convolution
            nn.Conv2d(in_channels=nf1, out_channels=nf2, kernel_size=ks),
            # Activation function
            nn.ReLU(),
            # 3rd convolution
            nn.Conv2d(in_channels=nf2, out_channels=nf3, kernel_size=ks),
            # Activation function
            nn.ReLU(),
            # 1st reduction
            nn.MaxPool2d(kernel_size=rd1),
            # 1st dropout
            nn.Dropout(p=dp1)
        )

        # Determine the height, width, and total size of the feature map
        feat_map_height = (self.dim_in - 6) // 2
        feat_map_width = (self.frame_width - 6) // 2
        self.conv_embedding_size = nf3 * feat_map_height * feat_map_width

        # Number of neurons for each fully-connected stage
        self.fc_embedding_size = 128 * self.model_complexity

        self.dense = nn.Sequential(
            # 1st fully-connected
            nn.Linear(in_features=self.conv_embedding_size,
                      out_features=self.fc_embedding_size),
            # Activation function
            nn.ReLU(),
            # 2nd dropout
            nn.Dropout(p=dp2,),
            # 2nd fully-connected
            nn.Linear(in_features=self.fc_embedding_size,
                      out_features=self.dim_out)
        )

    def forward(self, feats):
        '''
        Perform the main processing steps for TabCNN.

        Parameters
        ----------
        feats : Tensor (B x C x T x F)
          Input features for a batch of tracks,
          B - batch size
          C - number of channels in features
          T - number of frames
          F - number of features (frequency bins)

        Returns
        ----------
        output : Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        '''

        # Obtain the feature embeddings
        embeddings = self.conv(feats)
        # Flatten spatial features into one embedding
        embeddings = embeddings.flatten(1)
        # dense
        output = self.dense(embeddings)

        tablature = output.view(-1, self.num_strings, self.num_classes)
        # we don't softmax because we need to calculate loss in fit.
        # tablature = torch.softmax(tablature, dim=1)
        return tablature

    def fit(self, dataloader, optimizer):
        self.train(mode=True)
        running_loss = 0
        for batch in dataloader:
            features, targets = batch[0].to(self.device), batch[1].to(self.device)
            features = self.norm_fn(features)
            pred = self(features)

            if self.use_idle_stats is True:
                idle_stats = batch[2].to(self.device)
                sample_weights = self.sample_weight_fn(idle_stats)
            else:
                sample_weights = torch.ones(pred.shape[:2], device=self.device)
            if self.distance_weight_fn is not None:
                sample_weights *= self.distance_weight_fn(pred, targets)

            loss = multi_label_focal_loss(pred, targets, self.class_weights,
                                          sample_weights, self.gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(dataloader)
        return running_loss
    
    def evaluate(self, dataloader, label_metrics):
        self.train(mode=False)
        label_cms = dict()
        running_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                features, targets = batch[0].to(self.device), batch[1].to(self.device)
                features = self.norm_fn(features)
                pred = self(features)

                if self.use_idle_stats is True:
                    idle_stats = batch[2].to(self.device)
                    sample_weights = self.sample_weight_fn(idle_stats)
                else:
                    sample_weights = torch.ones(pred.shape[:2], device=self.device)
                if self.distance_weight_fn is not None:
                    sample_weights *= self.distance_weight_fn(pred, targets)

                label_cms = collect_loop_metrics(pred, targets, label_metrics, label_cms)
                running_loss += multi_label_focal_loss(pred, targets, self.class_weights,
                                          sample_weights, self.gamma).item()
        running_loss /= len(dataloader)
        return running_loss, label_cms
    
def collect_loop_metrics(pred, target, label_metrics, label_cms):
    for label in label_metrics:
        label_fn = label_metrics[label][0]
        micro_avg = label_metrics[label][1]
        y_hat, y = label_fn(pred), label_fn(target)
        cm = confusion_matrix(y_hat, y, micro_avg=micro_avg, flatten=True)
        if label not in label_cms:
            label_cms[label] = cm
        else:
            label_cms[label] += cm
    return label_cms