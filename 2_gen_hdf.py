from data_handling.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(buffer=0.5, cut_off=1)
datasets_list = ['GuitarTECHS', 'IDMT', 'EGDB', 'GuitarSet', 'EGSet12',
                 'GOAT', 'PLUS', 'SynthTab']
    
for dataset in datasets_list:
    print(dataset)
    preprocessor.trim = True if dataset in ['SynthTab', 'PLUS'] else False
    preprocessor.main(dataset)
preprocessor.generate_metadata()