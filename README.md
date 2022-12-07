# Mokapot Comparison based on different machine learning models

---

## Introduction

Mokapot enhances peptide detection by rescoring PSMs, similar to Percolator, although despite the percolator it allows users to apply different machine learning models instead of using only linear support vector machine classification. 

By using nonlinear classification models, Mokapot may identify more PSMs but how many of them are true? 

We added pyrococcus as entrapment in our data to check the ratio of false negatives or entrapments to increased identified. Various machine learning models can be used in this repo by simply choosing the model in main file among:
```
'default', 'random_forest', 'xgboost', 'svc','logistic_regression', 'knn', 'mlp', 'gradient_boost'
 ```
 
 Tuned models and their dataframe results will be saved in 
```
./saved_models/
```
