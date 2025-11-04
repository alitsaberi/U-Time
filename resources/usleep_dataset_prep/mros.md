## Dataset: MROS

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/mros/polysomnography/edfs/visit*/*.edf" --out_dir "$ROOT_PATH/raw/mros/" --resample 128 --channels C3-M2 C4-M1 E1-M2 E2-M1
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/mros/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/mros/" --resample 128 --channels ECG_L ECG_R --rename ECG1 ECG2
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/mros/polysomnography/annotations-events-nsrr/visit*/*.xml" --out_dir "$ROOT_PATH/raw/mros/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/mros/" --subject_dir_pattern 'mros*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)'
```

Notes:
- 2 visits: 'visit1', 'visit2'
- example nameing: mros-visit1-aa5665, mros-visit2-aa5665
- match regex: .*?-.*?-(.*)
