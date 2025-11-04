## Dataset: SEDF-SC

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/sleep-edfx/sleep-cassette/*PSG.edf" --out_dir "$ROOT_PATH/raw/sedf-sc/" --resample 128 --channels 'EEG Fpz-Cz' 'EEG Pz-Oz' 'EOG horizontal' --rename Fpz-Cz Pz-Oz EOG
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/sleep-edf-extended/SC*/*Hypnogram.edf" --out_dir "$ROOT_PATH/raw/sedf-sc/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/sedf-sc/" --subject_dir_pattern 'SC*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex 'SC4(\d{2}).*'
```

Notes: 
- Two sleep studies made on each subject (The first nights of subjects 36 and 52, and the second night of subject 13, were lost due to a failing cassette or laserdisk.)
- 2 groups, Files are named in the form SC4ssNEO-PSG.edf where 'ss' is the subject number, and 'N' is the night.
- example nameing: 'SC4801G0-PSG', 'SC4802G0-PSG' 
- match regex: 'SC4(\d{2}).*'
