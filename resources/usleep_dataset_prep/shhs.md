## Dataset: SHHS

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/shhs/polysomnography/edfs/shhs*/*.edf" --out_dir "$ROOT_PATH/raw/shhs/" --resample 128 --channels 'EEG' 'EEG(sec)' 'EOG(L)' 'EOG(R)'  --rename_channels 'C4-A1' 'C3-A2' 'EOG(L)-PG1' 'EOG(R)-PG1'
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/shhs/polysomnography/edfs/shhs*/*.edf" --out_dir "$ROOT_PATH/raw/shhs/" --resample 128 --channels ECG
```

#### Extract hypno command
```
ut extract_hypno --file_pattern "$ROOT_PATH/source/shhs/polysomnography/annotations-events-nsrr/shhs*/*.xml" --out_dir "$ROOT_PATH/raw/shhs/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/shhs/" --subject_dir_pattern 'shhs*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-(.*)'
```

Notes: 
- 2 visits: 'shhs1', 'shhs2'
- example nameing: 'shhs1-205238', 'shhs2-205238'
- match regex: .*?-(.*)
