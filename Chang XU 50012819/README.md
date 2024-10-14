# HKUSTGZ-AIAA-2205-HW1-Fall-2024
Chang XU 50012819
## Package Required
```
scikit-learn pandas tqdm numpy xgboost
```
## Select Frames
Choose 50 percent of the total mfccs.
```
python select_frames.py labels/trainval.csv 0.5 selected.mfcc.csv --mfcc_path mfcc/
```
## Train k-means model
As shown in the one-page-write-up, the choice of k = 430 is the most suitable for this task.
```
python train_kmeans.py selected.mfcc.csv 430 kmeans.430.model
```
## Feature Extraction
```
python get_bof.py kmeans.430.model 430 labels/videos.name.lst --mfcc_path mfcc/ --output_path bof430/
```
## Train XGBoost Model
```
python train_XGBoost.py bof430/ 430 labels/trainval.csv models/mfcc-430.XGBoost.model
```
## Use XGBoost Model to predict
```
python test_XGBoost.py models/mfcc-430.XGBoost.model bof430 430 labels/test_for_student.label mfcc-430.XGBoost.csv
```
## Validation programs
In this project, I use k-fold cross validation and stratified cross validation, for mlp model test and XGBoost method validation respectively. For mlp validation, the following command are used:
```
python Cross_Validation_MLP.py bof100/ 100 labels/trainval.csv models/mfcc-100.mlp.model
```
Or one can change the numbers in "bof100", "100", and "models/mfcc-100.mlp.model" for different k.
For XGBoost validation, use the following cammand:
```
python Stratified_Cross_Validation.py bof430/ 430 labels/trainval.csv models/mfcc-430.XGBoost.mlp.model
```
Also change the corresponding numbers is acceptable.
