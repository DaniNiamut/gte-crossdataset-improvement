import re
import os
import jams
import shutil
from utils.misc import *
from utils.utils import error_handling_makedirs, get_audio_annot_folder_path

def make_clean_folder(file_exceptions, ext_exceptions, dataset_exceptions,
                      folder_exceptions, accepted_audio_formats,
                      use_subfolder_suffix=False, use_deep_prefix=False):
    
    def files_filter(string_path):
        switch = True
        if string_path in file_exceptions:
            switch = False
        for exception in ext_exceptions:
            ext = os.path.splitext(string_path)
            if exception == ext[0]:
                switch = False
            elif exception == ext[1]:
                switch = False
        return switch

    for dataset in os.listdir(UNCLEAN_DATASET_PATH):
        if dataset == 'readme.md' or dataset in dataset_exceptions:
            continue
        
        # Make new paths for Clean Datasets folder.
        dataset_path = UNCLEAN_DATASET_PATH + '/' + dataset
        new_dataset_path, annot_path, audio_path = get_audio_annot_folder_path(dataset)
        for path in [new_dataset_path, annot_path, audio_path]:
            error_handling_makedirs(path)
        
        # Get all files across all valid subfolders
        roots, files = [], []
        for root, dirs, os_walk_files in os.walk(dataset_path):
            dirs[:] = [d for d in dirs if d not in folder_exceptions]

            files_in_dir = list(filter(files_filter, os_walk_files))
            files.extend(files_in_dir)
            roots.extend([root] * len(files_in_dir))

        # Unique tuple of origin and file names to prevent loop count bleeding
        paths = list(set(zip(roots, files)))
        
        # Check if duplicates exist within THIS specific dataset iteration
        actual_files = [p[1] for p in paths]
        has_duplicates = len(set(actual_files)) != len(actual_files)

        if has_duplicates:
            print(f'Duplicates found in {dataset}, applying modifications to file names.')

        count = 0
        for (root, old_file) in paths:
            # Force all slashes to forward slashes to keep splitting uniform
            old_path = os.path.normpath(root + '/' + old_file).replace('\\', '/')
            
            prefix, suffix = '', ''
            if use_subfolder_suffix or use_deep_prefix:
                relative_path = old_path.replace(dataset_path + '/', '')
                path_parts = relative_path.split('/')[:-1]
                if len(path_parts) >= 2 and use_deep_prefix:
                    prefix += f'{path_parts[-1]}_'
                if len(path_parts) >= 1 and use_subfolder_suffix:
                    suffix = f'_{path_parts[0]}'
            # Separate file name and extension BEFORE renaming
            base_name, ext = os.path.splitext(old_file)

            if has_duplicates:
                new_file = f"{count}~{prefix}{base_name}{suffix}{ext}"
            else:
                new_file = old_file
            
            if ext in accepted_audio_formats:
                shutil.copy2(old_path, audio_path + '/' + new_file)
            else:
                shutil.copy2(old_path, annot_path + '/' + new_file)
            count += 1