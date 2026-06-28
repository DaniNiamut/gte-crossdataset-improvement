# 2026 - TabCNN - Internship Dani

This repository contains the following:
* Annotation loading functions for `.gp5`, `.jams`, `.xml`, and `.midi` files into a standard format.
* Scripts to create a synthetic dataset (named PLUS) given annotations and chromatic scale recordings.
* Script to standardize a dataset folder structure.
* Scripts to preprocess datasets into an HDF5 file consisting of frame-level tablature and CQT-spectograms.
* TabCNN training pipeline with multiple configurable parameters.
* Functions to plot test results from TabCNN training.

## Using This Repo

First, you must download one or more of the following datasets supported:
* [EGDB](https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_)
* [EGSet12](https://zenodo.org/records/11406378)
* [GOAT](https://zenodo.org/records/15690894) 
* [GuitarSet](https://guitarset.weebly.com/)
* [GuitarTECHS](https://zenodo.org/records/14963133)
* [IDMT](https://zenodo.org/records/7544110) (Specifically Set 2 and Set 3).
* [SynthTab](https://rochester.app.box.com/v/SynthTab-Dev) (Specifically a 15,000 audio partition of it).

The downloaded dataset(s) need to be in one directory. You may change this directory to your liking with `CLEAN_DATASET_PATH` found in `\utils\misc.py`. This folder should also contain the necessary audio files used as input for the PLUS audio generation method under the `chromatic_scales/items` subfolder, which should contain more subfolders, each with 6 `.WAV` files corresponding to each string and 1 `metadata.json` file containing the highest fret and tempo. The dataset(s) must conform to the necessary file structure format seen below:
* 📁 **Datasets**
  * <details>
    <summary>📁 <b>dataset_GuitarSet</b></summary>
    <ul>
      <li>
        <details>
          <summary>📁 <b>annotations_tab</b></summary>
          <ul>
            <li>📄 `00_BN1-129-Eb_comp.jams`</li>
            <li>...</li>
            <li>📄 `05_SS3-98-C_solo.jams`</li>
          </ul>
        </details>
      </li>
      <li>
        <details>
          <summary>📁 <b>audio</b> <i></i></summary>
          <ul>
            <li>📄 `00_BN1-129-Eb_comp_hex.wav`</li>
            <li>...</li>
            <li>📄 `05_SS3-98-C_solo_hex.wav`</li>
          </ul>
        </details>
      </li>
    </ul>
  * ...  
  * <details>
    <summary>📁 <b>dataset_EGDB</b></summary>
    <ul>
      <li>
        <details>
          <summary>📁 <b>annotations_tab</b></summary>
          <ul>
            <li>📄 `00_BN1-129-Eb_comp.jams`</li>
            <li>...</li>
            <li>📄 `05_SS3-98-C_solo.jams`</li>
          </ul>
        </details>
      </li>
      <li>
        <details>
          <summary>📁 <b>audio</b> <i></i></summary>
          <ul>
            <li>📄 `0~188_audio_DI.wav`</li>
            <li>...</li>
            <li>📄 `1689~4_audio_Plexi.wav`</li>
          </ul>
        </details>
      </li>
    </ul>
  * <details>
    <summary>📁 <b>dataset_DadaGP</b></summary>
    <ul>
      <li>
        <details>
          <summary>📁 <b>annotations_tab</b></summary>
          <ul>
            <li>📄 `394~Rolling Stones (The) - Paint It, Black.gp3 - 2.jams`</li>
            <li>...</li>
            <li>📄 `81479~Kernundrum - Racked Off.gp3 - 2.jams`</li>
          </ul>
        </details>
      </li>
    </ul>
    
  </details>
  
You may alter the directory structure manually or use the function provided in `\utils\folder_cleaner.py`. In case of duplicate file names when using `make_clean_folder`, you may then also further specify to use an index to prevent overwriting or a prefix and or suffix based on the directory structure to distinguish between duplicates. These were considered mandatory for SynthTab and EGDB for example. 

Once the datasets have been downloaded, `DATASET_AFFIX_DICT` in `\utils\misc.py` and `dataset_picker_dict` in `\data_handling\picker_functions.py` must be checked. `DATASET_AFFIX_DICT` is a dictionary, where each key is a dataset name and the value is a regex pattern which is used in `file_grouper` (found in `\data_handling\load.py`) to group annotation files to audio files. Each of the adopted datasets use a different regex pattern due to their different naming schemes. In some cases, multiple audio files can be traced back to a single annotation or vice-versa. Due to this it is necessary to specify which annotation or audio file is picked in pre-processing. `dataset_picker_dict` is a dictionary where each key is the dictionary name and the value is a list of four items. The first two items need to be functions, which take a list of strings as input and output either 1 string (for the annotation) or a list of strings (for the audios). The last two items are `kwargs` for the respective functions. A handful of functions have been defined, but more may be easily added if necessary. This is illustrated in the example below:
```
dataset_picker_dict = { 'EGDB':[take_first_path, contains_substring,
        dict(), {'substring':'_DI', 'return_as_list':True}],}
```
It must be noted that this repository includes a base `DATASET_AFFIX_DICT` and `dataset_picker_dict`. These support the aforementioned datasets with three caveats. Firstly, When EGDB and GuitarTECHS were downloaded it was converted to the aforementioned directory structure using `make_clean_folder` with altered names to account for duplicate file names. This is reflected in `DATASET_AFFIX_DICT` and may not work on EGDB or GuitarTECHS if converted through a different method than `make_clean_folder`. Secondly, `dataset_picker_dict` is currently set to choose the DI audio and not any other audio. Thirdly, this repository does not fully support SynthTab. The regex pattern in `DATASET_AFFIX_DICT` for SynthTab  and `make_clean_folder` were proven to work for a few partitions of SynthTab, but have not been tested on the entire dataset yet. It is possible that either may break on some audios or annotations in SynthTab. This can be seen if using the SynthTab developer set, where `DATASET_AFFIX_DICT` will not work. Dealing with any issues in this regard should hopefully be manageable using the configurable regex patterns and picker functions.

## Creating PLUS

Once the previous items have been taken care of, the preprocessing or PLUS creation may take place. To generate PLUS, we must specify from where to draw the guitar sources and from which dataset to draw the annotations which will be overlayed. The guitar sources must also match a specific format. The path of the guitar sources may be specified in `\utils\misc.py` with `CHROM_SCALES_PATH`. The `1_gen_plus.py` script uses DadaGP by default. If using the default case, DadaGP must be downloaded too and put in the same directory as the other datasets with each annotation being in the `annotations_tab` folder as well. GuitarSet, SynthTab or EGDB may also be used instead if so desired.

## Preprocessing

As long as everything before the previous section has been completed, preprocessing should be straightforward. Running the `2_gen_hdf.py` script creates an HDF5 file with CQT-spectograms, tablature and frame--level metadata. used in training TabCNN later. By default, SynthTab and PLUS are trimmed to cut off excess silences of 1 second in duration or more. This may be turned off or extended to the other datasets. A metadata `.CSV` file is also generated. It is necessary for training and contains audio-level information per dataset.

## Training

Once the HDF5 file and the metadata `.CSV` file have been created, training may take place. Currently only the TabCNN model is defined. Training is done through 'experiments' defined in `experiments.py`. It is also possible to define a custom pipeline or a simpler training script. Each experiment in `experiments.py` can run a cross fold validation experiment and a simpler training run. There are many configurable options when defining an experiment.

