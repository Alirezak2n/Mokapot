parameters_gboost = {
    'max_depth': [9, 10, 15, 20],
    'min_samples_leaf': [0.001, 0.01, 0.1, 1.0, 5.0],
    "learning_rate": [0.1, 0.2, 1],
    'n_estimators': [200, 250, 300, 350, 400, 450, 500, 600],
    'criterion': ['friedman_mse']
}

parameters_nn = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 1e-2, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [100, 200, 500]
}

parameters_knn = {
    'leaf_size': [1, 3, 5, 10, 20, 50],
    'n_neighbors': [10, 20, 50],
    'p': [1, 2]
}

parameters_lr = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'C': [100, 10, 1.0, 0.1, 0.01]
}

parameters_svc = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

parameters_lsvc = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear']
}

parameters_rf = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_leaf': [0.001, 0.01, 0.1, 1.0, 5.0],
    'n_estimators': [100, 120, 150, 200, 400]
}

parameters_xgboost = {
    'max_depth': [3, 5, 6, 10, 15, 20],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.4, 1.0, 0.1),
    'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
    'min_samples_leaf': [0.001, 0.01, 0.1, 1.0, 5.0],
    'n_estimators': [100, 200, 300, 400]
}

dict_to_params = {"random_forest" : parameters_rf,
"xgboost" : parameters_xgboost,
"svc" : parameters_svc,
"logistic_regression" : parameters_lr,
"knn" : parameters_knn,
 "mlp" : parameters_nn,
 "gradient_boost" : parameters_gboost}