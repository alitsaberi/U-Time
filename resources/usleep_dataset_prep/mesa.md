## Dataset: MESA

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/mesa/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/mesa/" --resample 128 --channels EEG1 EEG2 EEG3 EOG-L EOG-R --rename Fz-Cz Cz-Oz C4-M1 E1-FPz E2-FPz
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/mesa/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/mesa/" --resample 128 --channels EKG PLETH --rename ECG PPG
```

#### Extract hypno command
```
ut extract_hypno --file_pattern "$ROOT_PATH/source/mesa/polysomnography/annotations-events-nsrr/*.xml" --out_dir "$ROOT_PATH/raw/mesa/" --id_regex "(.*)-nsrr"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/mesa/" --subject_dir_pattern 'mesa*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes: 
- No mentioned subject relations
- 1 group: 'sleep'
- example nameing: 'mesa-sleep-5805'
- match regex: None
