import os

UNCLEAN_DATASET_PATH = 'c:/Unclean_Datasets'
CLEAN_DATASET_PATH = 'c:/Datasets'
SPEC_AND_LABEL_PATH = os.getcwd() + '/CodeOutput/Spec&Label'
SYNTHTAB_JSON_PATH  =os.getcwd() + '/utils/note_tab.json'
VIS_PATH = os.getcwd() + '/CodeOutput/Visualizations'
PREV_MODELS_PATH = os.getcwd() + '/CodeOutput/PreviousModels'
CHROM_SCALES_PATH = CLEAN_DATASET_PATH + 'chromatic_scales'
MIDI_NOTE_NUMBERS = [40, 45, 50, 55, 59, 64]
CHROME_SCALES_STRING_NAMES = ['Low E', 'A', 'D', 'G', 'B', 'High E']
QUARTER_TIME = 960
STRING_NAME_TO_MIDI = {
    'Low E': 40, 
    'A': 45, 
    'D': 50,
    'G': 55,
    'B': 59,
    'High E': 64
}
DATASET_AFFIX_DICT = {
    'EGDB':r'(~clip\d+|~\d+)(?!\d)',
    'EGSet12':r'\w+',
    'GOAT':r'item_\d{1,3}',
    'GuitarSet':r'[\#a-zA-Z0-9_-]{0,16}[a-zA-Z]{0,4}',
    'GuitarTECHS': r'(?<=~midi_)[A-Za-z0-9_]+(?=\.[A-Za-z0-9]+$)|(?<=~exo_)[A-Za-z0-9_]+(?=\.[A-Za-z0-9]+$)|(?<=~ego_)[A-Za-z0-9_]+(?=\.[A-Za-z0-9]+$)|(?<=~micamp_)[A-Za-z0-9_]+(?=\.[A-Za-z0-9]+$)|(?<=~directinput_)[A-Za-z0-9_]+(?=\.[A-Za-z0-9]+$)',
    'IDMT':r'[^/]+?(?=VSBHD\.[a-zA-Z0-9]+$)|[^/]+?(?=\.[a-zA-Z0-9]+$)',
    'SynthTab': r'[a-zA-Z_\(\)\,\'\!\? ][a-zA-Z0-9_\(\)\,\'\!\? \-]+-*[`a-zA-Z0-9_\{\}\@\'\^\#\~\&\(\)\;\,\'\!\?\[\]\=\% \-]+ - gp\d+(?:__\d+)?',
    'PLUS': r'[^/\\]+(?=\.[^.]+$)',
    'DadaGP': r'~(.+)\.[^.]+$',
    }
