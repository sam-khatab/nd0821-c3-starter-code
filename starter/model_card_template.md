# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Model type: `RandomForestClassifier`
- Saved artifacts:
	- `model/rf_model.joblib`: trained classifier
	- `model/encoder.joblib`: fitted `OneHotEncoder`
	- `model/lb.joblib`: fitted `LabelBinarizer`
- Training libraries: scikit-learn, pandas, joblib
- Raw feature count: 14 input features
- Encoded feature count: 108 features after one-hot encoding
- Categorical features encoded: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- Continuous features passed through: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- Output labels stored in the saved label binarizer: `<=50K`, `>50K`
- Saved model parameters reflect scikit-learn defaults, including `n_estimators=100`, `criterion='gini'`, `max_features='sqrt'`, `bootstrap=True`, and `random_state=None`

## Intended Use

This model is intended to predict whether a person's annual income is less than or equal to $50K or greater than $50K using structured census-style demographic and employment features.


## Training Data

- Dataset source: `data/census.csv`
- Dataset size: 32,561 rows and 15 columns, including the target column `salary`
- Target label distribution:
	- `<=50K`: 24,720 rows
	- `>50K`: 7,841 rows
- The training script performs an 80/20 random train-test split
- Categorical features are one-hot encoded with `handle_unknown='ignore'`
- The target label is encoded with `LabelBinarizer`



## Evaluation Data

The training code evaluates the model on the 20% holdout portion produced by `train_test_split(data, test_size=0.20)`.


## Metrics

Metrics calculated from trained model:

Precision: 0.7274
Recall: 0.6157
F1: 0.6669



## Ethical Considerations

This model uses features such as race, sex, marital status, and native country. These are sensitive features or strong proxies for protected characteristics, so the model carries significant fairness and bias risk.

Potential risks include:

- Replicating historical bias present in the dataset
- Unequal error rates across demographic subgroups
- Disparate treatment or disparate impact if used in real decisions
- Misuse of a classroom/demo model in real-world decision pipelines

The model should not be used as the sole basis for consequential decisions about individuals.

## Caveats and Recommendations

- The saved model was trained with default random forest hyperparameters and no persisted tuning record
- The training script does not set a fixed `random_state`, so retraining is not reproducible by default
- Exact evaluation metrics for the currently saved artifact were not stored
- Inference depends on using the saved encoder and label binarizer together with the saved model

