import entrapment_counter
import pandas as pd
import mokapot
import random
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import logging_setup
import logging
from pathlib import Path

logging.getLogger("mokapot").setLevel(logging.WARNING)
logger = logging_setup.main(file_name='entrapment_logs.log')


def random_search(model, params, number_iterations=20, cross_validation=None, verbose=5, random_state=42,
                  number_jobs=15):
    model_obj = model()
    rs = RandomizedSearchCV(
        estimator=model_obj,
        param_distributions=params,
        scoring="roc_auc",
        n_iter=number_iterations,
        random_state=random_state,
        n_jobs=number_jobs,
        verbose=verbose,
        cv=cross_validation
    )
    return rs


def gboost():
    parameters = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
        'min_samples_leaf': [1],
        'min_samples_split': [2, 10, 100, 200, 400],
        "learning_rate": [0.1, 0.2, 1],
        'n_estimators': [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600],
        'criterion': ['friedman_mse']
    }

    ml_model = random_search(model=GradientBoostingClassifier, params=parameters)

    return ml_model


def mlp():
    parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 1e-2, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [100, 200, 500]
    }

    ml_model = random_search(model=MLPClassifier, params=parameters)

    return ml_model


def knn():
    parameters = {
        'leaf_size': [1, 3, 5, 10, 20, 50],
        'n_neighbors': [1, 10, 20, 50],
        'p': [1, 2]
    }

    ml_model = random_search(model=KNeighborsClassifier, params=parameters)

    return ml_model


def logistic_regression():
    parameters = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }

    ml_model = random_search(model=LogisticRegression, params=parameters)

    return ml_model


def percolator_model(train_subset=500000):
    perco = mokapot.PercolatorModel(subset_max_train=train_subset)
    return perco


def svc():
    # Support vector classification
    parameters = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    ml_model = random_search(model=SVC, params=parameters)

    return ml_model


def random_forest():
    # Random Forest tuning
    parameters = {
        'max_depth': [10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [10, 20, 40, 80, 100, 120, 150, 200, 400]}

    ml_model = random_search(model=RandomForestClassifier, params=parameters)

    return ml_model


def xgboost():
    parameters = {
        'max_depth': [3, 5, 6, 10, 15, 20],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
        'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
        'n_estimators': [100, 200, 300, 400]
    }
    ml_model = random_search(model=xgb.XGBClassifier, params=parameters)

    return ml_model


def mokapot_and_entrapment(model, ml_model, psms):
    result = []
    start_time = time.time()

    if model == 'default':
        model_obj = ml_model
    else:
        model_obj = mokapot.model.Model(ml_model)

    results, mokapot_models = mokapot.brew(psms, model=model_obj, max_workers=64)
    logger.info(f"--Mokapot finished in {time.time() - start_time :.2f} seconds--")

    folder_time = time.strftime("%Y%m%d")
    file_time = time.strftime("%H%M")
    Path('./saved_models/' + folder_time).mkdir(parents=True, exist_ok=True)
    mokapot_models[0].save(f'./saved_models/{folder_time}/{model}_{file_time}.pickle')
    results.to_txt(file_root=f'./saved_models/{folder_time}/{model}_{file_time}')

    logger.info("--Starting entrapment counter--")
    start_time = time.time()
    entraps = entrapment_counter.main(psms_table=results.confidence_estimates['psms'])
    logger.info(f"--Entrapment counting finished in  {time.time() - start_time :.2f} seconds--")

    result.append((results.accepted['psms'], results.accepted['peptides'], entraps))
    logger.info(result)


def main(model='default'):
    models = {
        'default': percolator_model, 'random_forest': random_forest, 'xgboost': xgboost, 'svc': svc,
        'logistic_regression': logistic_regression, 'knn': knn, 'mlp': mlp, 'gradient_boost': gboost
    }
    chosen_model = models[model]

    logger.info("--Reading the pin file--")
    start_time = time.time()
    psms = mokapot.read_pin('./data/PXD006932/txt/msms_searchengine_ms2pip_rt_features.pin')
    logger.info(f"--The pin file is read in {time.time() - start_time :.2f} seconds--")
    logger.info(f"--Starting Mokapot with {model} model--")
    ml_model = chosen_model()
    mokapot_and_entrapment(model, ml_model, psms)


def svc_test():
    result = []
    random_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    logger.info("--Reading the pin file--")
    start_time = time.time()
    psms = mokapot.read_pin('./data/PXD006932/txt/msms_searchengine_ms2pip_rt_features.pin')
    logger.info(f"--The pin file is read in {time.time() - start_time :.2f} seconds--")
    for i in range(10):
        l = {}
        for k, v in random_grid.items():
            ran = random.choice(random_grid[k])
            l[k] = ran
        rf = SVC(**l)
        mcls = mokapot.model.Model(rf)
        logger.info("--Starting Mokapot with SVC model--")
        results, models = mokapot.brew(psms, model=mcls, max_workers=64)
        logger.info(f"--Mokapot finished in {time.time() - start_time :.2f} seconds--")

        logger.info("--Starting entrapment counter--")
        start_time = time.time()
        entraps = entrapment_counter.main(psms_table=results.confidence_estimates['psms'])
        logger.info(f"--Entrapment counting finished in  {time.time() - start_time :.2f} seconds--")

        result.append((l, results.accepted['psms'], results.accepted['peptides'], entraps))
        logger.info(result)


if __name__ == '__main__':  # 'default','random_forest','xgboost','svc'
    # main('random_forest')
    main('default')
    # test()
