import argparse
import deepchem as dc
import joblib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

cached_datasets = {}


def load_datasets(featurizer):
    if featurizer in cached_datasets:
        return cached_datasets[featurizer]

    # Load the entire MUV dataset.
    print(f'Loading the MUV dataset with featurizer {featurizer}.')
    tasks, (dataset,), transformers = dc.molnet.load_muv(featurizer=featurizer, splitter=None)

    # Split the dataset into 5 folds.
    print('Splitting the dataset into 5 folds.')
    splitter = dc.splits.RandomStratifiedSplitter()
    datasets = splitter.k_fold_split(dataset, 5, seed=826)

    # Cache the dataset for future use.
    cached_datasets[featurizer] = datasets

    return tasks, datasets


def extract_task(dataset, task):
    return dc.data.NumpyDataset(dataset.X, dataset.y[:, task], dataset.w[:, task], dataset.ids)


def evaluate_model(tasks, datasets, model_generator, n_jobs=-1):
    scores = np.zeros((len(datasets), len(tasks)))
    roc_auc_score = dc.metrics.Metric(dc.metrics.roc_auc_score)

    def evaluate_model_one(fold, task):
        print(f'Evaluating on fold {fold} and task {task}.')
        model = model_generator()
        model.fit(extract_task(datasets[fold][0], task))
        scores[fold, task] = model.evaluate(extract_task(datasets[fold][1], task), [roc_auc_score])['roc_auc_score']

    joblib.Parallel(n_jobs=n_jobs, backend='threading')(
        joblib.delayed(evaluate_model_one)(fold, task)
        for fold in range(len(datasets))
        for task in range(len(tasks)))

    return scores


def logistic_regression_model():
    tasks, datasets = load_datasets('ecfp')

    print('Evaluating logistic regression model.')
    scores = evaluate_model(tasks, datasets, lambda: dc.models.SklearnModel(LogisticRegression()), n_jobs=1)
    pd.DataFrame(scores, columns=tasks).to_csv('results/lr.csv')


def random_forest_model():
    tasks, datasets = load_datasets('ecfp')

    for n_estimators in [1000, 4000, 16000]:
        for max_features in [None, 'sqrt', 'log2']:
            for class_weight in [None, 'balanced', 'balanced_subsample']:
                model_args = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'class_weight': class_weight
                }
                print(f'Evaluating random forest model with {model_args}.')
                scores = evaluate_model(tasks, datasets, lambda: dc.models.SklearnModel(
                    RandomForestClassifier(**model_args, n_jobs=1)))
                pd.DataFrame(scores, columns=tasks).to_csv(
                    f'results/rf_{n_estimators}_{max_features}_{class_weight}.csv')


def graph_convolution_model():
    tasks, datasets = load_datasets('graphconv')

    for layers in [[64, 64, 128], [128, 128, 256], [256, 256, 512]]:
        for dropout in [0.0, 0.1, 0.2]:
            model_args = {
                'n_tasks': 1,
                'graph_conv_layers': layers[:-1],
                'dense_layer_size': layers[-1],
                'dropout': dropout
            }
            print(f'Evaluating graph convolution model with {model_args}.')
            scores = evaluate_model(tasks, datasets, lambda: dc.models.GraphConvModel(**model_args))
            pd.DataFrame(scores, columns=tasks).to_csv(f'results/gc_{"_".join(map(str, layers))}_{dropout}.csv')


if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='BMI 826 Project - Kevin Gaffney')
    parser.add_argument('--lr', dest='lr_flag', action='store_true', help='run logistic regression model')
    parser.add_argument('--rf', dest='rf_flag', action='store_true', help='run random forest model')
    parser.add_argument('--gc', dest='gc_flag', action='store_true', help='run graph convolution model')
    args = parser.parse_args()

    if args.lr_flag:
        logistic_regression_model()

    if args.rf_flag:
        random_forest_model()

    if args.gc_flag:
        graph_convolution_model()
