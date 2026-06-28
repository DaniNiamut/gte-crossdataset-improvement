import json
import torch
import csv
import pickle
import librosa
import requests
import pandas as pd
from utils.misc import *

def get_audio_annot_folder_path(dataset):
    folder_path = CLEAN_DATASET_PATH + 'dataset_' + dataset
    annot_path = folder_path + '/annotations_tab'
    audio_path = folder_path + '/audio'
    return folder_path, annot_path, audio_path

def error_handling_makedirs(dir_path):
    '''
    Applies os.makedirs, but with error handling.
    Taken from https://www.geeksforgeeks.org/python/create-a-directory-in-python/
    '''
    try:
        os.makedirs(dir_path)
        print(f"Nested directories '{dir_path}' created successfully.")
    except FileExistsError:
        print(f"One or more directories in '{dir_path}' already exist.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def dict_from_lists(key_list, val_list):
    '''
    key_list[i] denotes which key val_list[i] pertains to.
    '''
    
    if len(val_list) != len(key_list):
        raise TypeError('Input lists must be of same length.')
    output_dict = {}
    for key in set(key_list):
        output_dict[key] = []
    for i in range(len(val_list)):
        output_dict[key_list[i]].append(val_list[i])
    return output_dict

def load_csv_as_dict(path):
    csvfile = open(path, newline='')
    csv_obj = csv.DictReader(csvfile)
    keys = []
    vals = []
    for row in csv_obj:
        keys.extend(list(row.keys()))
        vals.extend(list(row.values()))
    return dict_from_lists(keys, vals)

def midi_note_mode(list_vals):
    midi_vals = [round(librosa.hz_to_midi(val)) for val in list_vals]
    # Source - https://stackoverflow.com/a/28129716
    # Posted by David Dao, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-03-16, License - CC BY-SA 4.0
    midi_mode = max(set(midi_vals), key=midi_vals.count)
    return librosa.midi_to_hz(midi_mode)

def tukey_window(M, alpha=0.5, periodic=True, device: torch.device | None = None):
    r"""
    Taken from scipy.tukey to work with torch, see:
    https://github.com/scipy/scipy/blob/v1.17.0/scipy/signal/windows/_windows.py#L879-L964

    Return a Tukey window, also known as a tapered cosine window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero, an empty array
        is returned. An exception is thrown when it is negative.
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.
    periodic : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    device (torch.device, optional): the desired device of returned tensor. 
        Default: if None, uses the current device for the default tensor type
        (see torch.set_default_device). device will be the CPU for CPU tensor types 
        and the current CUDA device for CUDA tensor types.
    
    Returns
    -------
    w : torch.tensor
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    References
    ----------
    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
           Analysis with the Discrete Fourier Transform". Proceedings of the
           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Tukey_window

    Examples
    --------
    Plot the window and its frequency response:

    >>> import torch
    >>> import matplotlib.pyplot as plt

    >>> window = signal.windows.tukey(51)
    >>> plt.plot(window)
    >>> plt.title("Tukey window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.ylim([0, 1.1])
    """
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    if alpha <= 0:
        return torch.ones(M, dtype=torch.float64, device=device)
    elif alpha >= 1.0:
        return torch.hann_window(M, periodic=periodic, dtype=torch.float64, device=device)

    M, needs_trunc = (M, False) if periodic else (M + 1, True)

    n = torch.arange(0, M, dtype=torch.float64, device=device)
    width = torch.tensor(alpha*(M-1)/2.0, dtype=torch.float64, device=device)
    width = int(torch.floor(width))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0 * n1 / alpha /(M - 1))))
    w2 = torch.ones(n2.shape, device=device)
    w3 = 0.5 * (1 + torch.cos(torch.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1))))

    w = torch.concat((w1, w2, w3))
    return w[:-1] if needs_trunc else w

def log_compression_norm(x, alpha=10):
    log_x = torch.log(1 + alpha * x)
    return torch.nn.functional.normalize(log_x, p=2, dim=3)

def identity_function(input, **kwargs):
    '''
    The identity function. It seems that there is no built-in function 
    for it in Python, so I made it here. The use-case for this is when
    we want all audio.
    '''
    return input