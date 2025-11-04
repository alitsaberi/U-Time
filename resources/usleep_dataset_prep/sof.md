## Dataset: SOF

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/sof/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/sof/" --resample 128 --channels C3-A2 C4-A1 LOC-A2 ROC-A1
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/sof/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/sof/" --resample 128 --channels ECG1 ECG2
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/sof/polysomnography/annotations-events-nsrr/*.xml" --out_dir "$ROOT_PATH/raw/sof/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/sof/" --subject_dir_pattern 'sof*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No mentioned subject relations
- 1 group: 'visit8'
- example nameing: 'sof-visit-8-07853'
- match regex: None
