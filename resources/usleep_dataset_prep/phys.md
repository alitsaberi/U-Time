## Dataset: PHYS

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/phys/*/tr*/*.mat" --out_dir "$ROOT_PATH/raw/phys/" --resample 128 --use_dir_names --channels F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1 E1-M2
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/phys/*/tr*/*.mat" --out_dir "$ROOT_PATH/raw/phys/" --resample 128 --use_dir_names --channels ECG 
```

#### Extract hypno command
```
ut extract_hypno --file_pattern "$ROOT_PATH/source/phys/*/tr*/*HYP.ids" --out_dir "$ROOT_PATH/raw/phys/" --use_dir_names
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/phys/" --subject_dir_pattern 'tr*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100
```

Notes:
- No subject relations specified
- example nameing: 'tr13-0566'
- match regex: None
