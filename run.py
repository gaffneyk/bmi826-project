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
    cached_datasets[featurizer] = (tasks, datasets)

    return tasks, datasets


def extract_task(dataset, task):
    return dc.data.NumpyDataset(dataset.X, dataset.y[:, task], dataset.w[:, task], dataset.ids)


def evaluate_model(name, tasks, datasets, model_generator, n_jobs=-1):
    metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score),
               dc.metrics.Metric(dc.metrics.prc_auc_score)]

    roc_scores = np.zeros((len(datasets), len(tasks)))
    prc_scores = np.zeros((len(datasets), len(tasks)))

    def evaluate_model_one(fold, task):
        print(f'Evaluating on fold {fold} and task {task}.')
        model = model_generator()
        model.fit(extract_task(datasets[fold][0], task))
        scores = model.evaluate(extract_task(datasets[fold][1], task), metrics)

        roc_scores[fold, task] = scores['roc_auc_score']
        prc_scores[fold, task] = scores['prc_auc_score']

    joblib.Parallel(n_jobs=n_jobs, backend='threading')(
        joblib.delayed(evaluate_model_one)(fold, task)
        for fold in range(len(datasets))
        for task in range(len(tasks)))

    pd.DataFrame(roc_scores, columns=tasks).to_csv(f'results/{name}_roc.csv')
    pd.DataFrame(prc_scores, columns=tasks).to_csv(f'results/{name}_prc.csv')


def logistic_regression_model():
    tasks, datasets = load_datasets('ecfp')

    print('Evaluating logistic regression model.')
    evaluate_model('lr', tasks, datasets, lambda: dc.models.SklearnModel(LogisticRegression()), n_jobs=1)


def random_forest_model():
    tasks, datasets = load_datasets('ecfp')

    for n_estimators in [1000, 4000, 16000]:
        for max_features in [None, 'sqrt', 'log2']:
            model_args = {
                'n_estimators': n_estimators,
                'max_features': max_features,
            }
            print(f'Evaluating random forest model with {model_args}.')
            evaluate_model(
                f'rf_{n_estimators}_{max_features}',
                tasks,
                datasets,
                lambda: dc.models.SklearnModel(RandomForestClassifier(**model_args, n_jobs=1))
            )


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
            evaluate_model(
                f'gc_{"_".join(map(str, layers))}_{dropout}',
                tasks,
                datasets,
                lambda: dc.models.GraphConvModel(**model_args),
                n_jobs=1
            )


def weave_model():
    tasks, datasets = load_datasets('weave')

    print('Evaluating weave model.')
    evaluate_model('wv', tasks, datasets, lambda: dc.models.WeaveModel(n_tasks=1, batch_normalize=False), n_jobs=1)


def performance_analysis():
    print('Loading the MUV dataset with featurizer weave.')
    tasks, datasets, transformers = dc.molnet.load_muv(featurizer='weave', splitter='stratified')
    train_dataset, valid_dataset, test_dataset = datasets
    model = dc.models.WeaveModel(n_tasks=1, batch_normalize=False)

    print('Evaluating weave model.')
    model.fit(extract_task(valid_dataset, 0))
    model.predict(extract_task(test_dataset, 0))


if __name__ == '__main__':
    Path('results').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='BMI 826 Project - Kevin Gaffney')
    parser.add_argument('--lr', dest='lr_flag', action='store_true', help='run logistic regression model')
    parser.add_argument('--rf', dest='rf_flag', action='store_true', help='run random forest model')
    parser.add_argument('--gc', dest='gc_flag', action='store_true', help='run graph convolution model')
    parser.add_argument('--wv', dest='wv_flag', action='store_true', help='run weave model')
    parser.add_argument('--perf', dest='perf_flag', action='store_true', help='run performance analysis')
    args = parser.parse_args()

    if args.lr_flag:
        logistic_regression_model()

    if args.rf_flag:
        random_forest_model()

    if args.gc_flag:
        graph_convolution_model()

    if args.wv_flag:
        weave_model()

    if args.perf_flag:
        performance_analysis()
