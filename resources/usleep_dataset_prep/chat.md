## Dataset: CHAT

#### Extract commands

##### Extract EEG and EOG channels
```
ut extract --file_pattern "$ROOT_PATH/source/chat/polysomnography/edfs/*/*.edf" --out_dir "$ROOT_PATH/raw/chat/" --resample 128 --channels F3-M2 F4-M1 C3-M2 C4-M1 T3-M2 T4-M1 O1-M2 O2-M1 E1-M2 E2-M1
```

##### Extract ECG and PPG channels
```
ut extract --file_pattern "$ROOT_PATH/source/chat/polysomnography/edfs/*.edf" --out_dir "$ROOT_PATH/raw/chat/" --resample 128 --channels ECG1 ECG2 ECG3 EKG PLETHW PLETHNELLCOR --rename ECG1 ECG2 ECG3 ECG1 PPG PPG
```

#### Extract hypno command
```
ut extract_hypno --file_regex "$ROOT_PATH/source/chat/polysomnography/annotations-events-nsrr/*/*.xml" --out_dir "$ROOT_PATH/raw/chat/"
```

#### Views command
```
ut cv_split --data_dir "$ROOT_PATH/processed/chat/" --subject_dir_pattern 'chat*' --CV 1 --validation_fraction 0.10 --max_validation_subjects 50 --test_fraction 0.15 --max_test_subjects 100 --subject_matching_regex '.*?-.*?-(.*)'
```

Notes: 
- 3 groups: 'baseline', 'followup', 'nonrandomized'
- example nameing: 'chat-baseline-300001', 'chat-followup-300001'
                  'chat-nonrandomized-300004'
- match regex: .*?-.*?-(.*)
