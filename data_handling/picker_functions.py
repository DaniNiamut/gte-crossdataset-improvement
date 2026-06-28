from utils.utils import identity_function
import random
'''
This file defines the functions to be used in the DataPreprocessor class, specifically 
the data_handler function. These functions pick the annotation or audio path which will 
be used. Most of these could be defined as lambda functions for brevity.
'''

def take_first_path(paths, return_as_list=False, **kwargs):
    '''
    To be used when there is only one path in paths or when there is 
    a legitimate reason to always choose the first path in the list
    '''
    if return_as_list:
        return [paths[0]]
    else:
        return paths[0]
    
def take_rand_path(paths, return_as_list=False, **kwargs):
    '''
    To be used when there is only one path in paths or when there is 
    a legitimate reason to always choose the first path in the list
    '''
    rand_idx = random.randint(0, len(paths) - 1)
    if return_as_list:
        return [paths[rand_idx]]
    else:
        return paths[rand_idx]

def contains_substring(paths, substring, return_as_list=False, reverse=False, **kwargs):
    '''
    Chooses the path which contains the substring. Use-case for this is seen in GOAT.
    One would prefer the fine aligned MIDI tabs if possible, but sometimes only one annotation
    is offered. In this case, the function returns the only offered tablature. This function will 
    cause an error if there are multiple paths
    '''
    if len(paths) == 1:
        return paths[0] if return_as_list is False else paths
    
    if reverse:
        substring_filter = lambda string: substring not in string
    else:
        substring_filter = lambda string: substring in string
    paths_with_substring = list(filter(substring_filter, paths))

    if len(paths_with_substring) == 0:
        TypeError(f'The chosen substring "{substring}" led to 0 paths being usable.')
    if return_as_list:
        return paths_with_substring
    else:
        if len(paths_with_substring) > 1:
            TypeError(f'The chosen {substring} was found in more than 1 path.')
        return paths_with_substring[0]
        
# Structure of dataset_picker_dict:
# Dataset name : [annot picker function, audio picker function,
#   additional arguments to annot_picker, additional arguments to audio_picker]
# Remarks: the picker functions are defined in picker_functions.py. Also, if no
#   additional arguments are required, then an empty dictionary is passed.
dataset_picker_dict = {
'EGDB':[take_first_path, contains_substring,
        dict(), {'substring':'_DI', 'return_as_list':True}],
'EGSet12':[take_first_path, take_first_path,
            dict(), {'return_as_list': True}],
'GOAT':[take_first_path, take_first_path,
        dict(), {'return_as_list':True}],
'GuitarSet':[take_first_path, take_first_path,
                dict(), {'return_as_list':True}],
'GuitarTECHS': [take_first_path, contains_substring,
                dict(), {'substring':'directinput', 'return_as_list':True}],
'IDMT':[take_first_path, take_first_path,
        dict(), {'return_as_list': True}],
'SynthTab':[take_first_path, take_rand_path,
            dict(), {'return_as_list': True}],
'PLUS':[take_first_path, identity_function,
        dict(), {'return_as_list': True}],
'DadaGP':[take_first_path, None,
        dict(), None]}