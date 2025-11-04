## Dataset: ISRUC-SG3

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/ISRUC/sg3*/*.edf" --out_dir "$ROOT_PATH/raw/isruc-sg3/" --resample 128 --use_dir_names --channels 'F3-M2' 'C3-M2' 'O1-M2' 'F4-M1' 'C4-M1' 'O2-M1' 'E1-M2' 'E2-M1'
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/ISRUC/sg3*/*_1-HYP.npz" --out_dir "$ROOT_PATH/raw/isruc-sg3/"
```

#### Views command
```
None (all test)
```

Notes: 
- No subject relations specified
- example nameing: 'sg3_subject_3'
- match regex: None
