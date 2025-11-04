## Dataset: ISRUC-SG1

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/ISRUC/subject*/*.edf" --out_dir "$ROOT_PATH/raw/isruc-sg1/" --resample 128 --use_dir_names --channels 'F3-M2' 'C3-M2' 'O1-M2' 'F4-M1' 'C4-M1' 'O2-M1' 'E1-M2' 'E2-M1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/ISRUC/subject_*/*.npz" --out_dir "$ROOT_PATH/raw/isruc-sg1/"
```

#### Views command
```
None (all test)
```

Notes: 
- No subject relations specified
- example nameing: 'subject_82'
- match regex: None
