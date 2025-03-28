## Dataset: Wearanize+

#### Export to USleep format

#### Views command
```
ut cv_split --data_dir [path_to_data_directory] --out_dir [path_to_output_directory] --subject_dir_pattern 'Sub*' --CV 1 --subject_matching_regex 'Sub(\d{3})' --validation_fraction 0.2 --test_fraction 0.0
```

Notes: 
- There is one recording per subject 
- example nameing: 'Sub001'
- Default match regex: 'Sub(\d{3})'
