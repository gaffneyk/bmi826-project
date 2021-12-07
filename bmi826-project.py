import deepchem as dc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import multiprocessing

# First, load the entire MUV dataset. This may take a few minutes.
tasks, (dataset,), transformers = dc.molnet.load_muv(featurizer=dc.feat.RawFeaturizer(smiles=True), splitter=None)
n_tasks = len(dataset.tasks)
n_folds = 5

# Then, split the dataset into training and test datasets, and further split the training dataset into 5 folds.
splitter = dc.splits.RandomStratifiedSplitter()
train_dataset, test_dataset = splitter.train_test_split(dataset, seed=826)
train_datasets = splitter.k_fold_split(train_dataset, n_folds, seed=826)


# Finally, featurize the dataset with circular fingerprint and graph convolution featurizers.
def featurize(featurizer, train_datasets, test_dataset):
    transformer = dc.trans.FeaturizationTransformer(featurizer=featurizer)

    test_dataset_featurized = transformer.transform(test_dataset)
    train_datasets_featurized = [
        (transformer.transform(train_dataset), transformer.transform(cv_dataset))
        for train_dataset, cv_dataset in train_datasets
    ]

    return train_datasets_featurized, test_dataset_featurized


ecfp_dataset = featurize(dc.feat.CircularFingerprint(), train_datasets, test_dataset)
graphconv_dataset = featurize(dc.feat.ConvMolFeaturizer(), train_datasets, test_dataset)
weave_dataset = featurize(dc.feat.WeaveFeaturizer(), train_datasets, test_dataset)


def extract_task(dataset, task):
    return dc.data.NumpyDataset(dataset.X, dataset.y[:, task], dataset.w[:, task], dataset.ids)


def evaluate(dataset, model_generator, model_args):
    scores = np.zeros((n_folds, n_tasks))
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    train_datasets, test_dataset = dataset

    for fold in range(n_folds):
        train_dataset, cv_dataset = train_datasets[fold]

        for task in range(n_tasks):
            train_dataset_task = extract_task(train_dataset, task)
            cv_dataset_task = extract_task(cv_dataset, task)

            m = model_generator(**model_args)
            m.fit(train_dataset_task)
            scores[fold, task] = m.evaluate(cv_dataset_task, [metric])['roc_auc_score']

    return scores


lr_scores = evaluate(ecfp_dataset, dc.models.SklearnModel, LogisticRegression())

rf_n_estimators = [1000, 4000, 16000]
rf_max_features = [None, 'sqrt', 'log2']
rf_class_weight = [None, 'balanced', 'balanced_subsample']

rf_model_args = [
    {
        'model': RandomForestClassifier(n_estimators=n_estimators,
                                        max_features=max_features,
                                        class_weight=class_weight)
    }
    for n_estimators in rf_n_estimators
    for max_features in rf_max_features
    for class_weight in rf_class_weight
]

rf_scores = [(model_args, evaluate(ecfp_dataset, dc.models.SklearnModel, model_args))
             for model_args in rf_model_args]

gc_model_args = [
    {
        'n_tasks': 1,
        'graph_conv_layers': layers[:-1],
        'dense_layer_size': layers[-1],
        'dropout': dropout
    }
    for layers in [[64, 64, 128], [128, 128, 256], [256, 256, 512]]
    for dropout in [0.0, 0.1, 0.2]
]

gc_scores = [(model_args, evaluate(graphconv_dataset, dc.models.GraphConvModel, model_args))
             for model_args in gc_model_args]
