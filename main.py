import entrapment_counter
import mokapot
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import logging_setup
import logging
from pathlib import Path

import search_params

logging.getLogger("mokapot").setLevel(logging.WARNING)
logger = logging_setup.main(file_name='new_params_logs.log')


def percolator_model(train_subset=500000):
    perco = mokapot.PercolatorModel(subset_max_train=train_subset)
    return perco


def mokapot_and_entrapment(
        model,
        ml_model,
        psms
):
    result = []
    start_time = time.time()

    if model == 'default':
       model_obj = ml_model
    else:
        model_obj = mokapot.model.Model(ml_model)

    results, mokapot_models = mokapot.brew(psms, model=model_obj, max_workers=64)
    logger.info(f"--Mokapot finished in {time.time() - start_time :.2f} seconds with {model}--")

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


def main(
        model='default',
        number_iterations=20,
        cross_validation=None,
        verbose=5,
        random_state=42,
        number_jobs=15,
        pin_file='./data/PXD006932/txt/msms_searchengine_ms2pip_rt_features.pin'
):
    models = {
        'default': percolator_model(),
        'random_forest': RandomForestClassifier(),
        'xgboost': xgb.XGBClassifier(),
        'svc': SVC(),
        'logistic_regression': LogisticRegression(),
        'knn': KNeighborsClassifier(),
        'mlp': MLPClassifier(),
        'gradient_boost': GradientBoostingClassifier()
    }

    logger.info("--Reading the pin file--")
    start_time = time.time()
    psms = mokapot.read_pin(pin_file)
    logger.info(f"--The pin file is read in {time.time() - start_time :.2f} seconds--")

    model_obj = models[model]

    if model != 'default':
        params = search_params.dict_to_params[model]

        # The min samples leaf
        if "min_samples_leaf" in params.keys():
            params['min_samples_leaf'] = [v * len(psms) for v in params['min_samples_leaf']]

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
    elif model == 'default':
        rs = model_obj
    logger.info(f"--Starting Mokapot with {model} model--")
    mokapot_and_entrapment(model, rs, psms)


if __name__ == '__main__':  # 'default','random_forest','xgboost','svc','logistic_regression','knn','mlp','gradient_boost'
    # main('random_forest')
    main('svc')
    # test()
