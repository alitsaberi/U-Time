## Dataset: HPAP

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/homepap/polysomnography/edfs/lab/*/*.edf" --out_dir "$ROOT_PATH/raw/homepap/" --resample 128 --channels F4-M1 C4-M1 O2-M1 C3-M2 F3-M2 O1-M2 E1-M2 E2-M1 E1 E2
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/homepap/polysomnography/edfs/lab/*/*.edf" --out_dir "$ROOT_PATH/raw/homepap/" --resample 128 --channels ECG1 ECG2 ECG3 ECG PLETH --rename ECG1 ECG2 ECG3 ECG1 PPG
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/homepap/polysomnography/annotations-events-nsrr/lab/*/*.xml" --out_dir "$ROOT_PATH/raw/hpap/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/hpap/" --subject_dir_pattern 'homepap*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*-(\d+)'
```

Notes: 
- No mentioned subject relations
- 2 group: 'full', 'split'
- example nameing: 'homepap-lab-full-1600039',
                  'homepap-lab-split-1600150'
- match regex (actually not be needed here): .*-(\d+)
