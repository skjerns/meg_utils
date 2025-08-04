# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:59:16 2024

@author: Simon Kern
"""
from pathlib import Path
import joblib
import os
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from tqdm import tqdm
import warnings
import inspect
import itertools
import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier
from sklearn.ensemble._voting import LabelEncoder, _routing_enabled
from sklearn.ensemble._voting import process_routing, Bunch
from sklearn.ensemble._voting import _fit_single_estimator

try:
    from . import misc
    from .misc import NumpyEncoder
except ImportError:
    import misc
    from misc import NumpyEncoder


def is_json_serializable(obj):
    """
    Check if an object is JSON serializable.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is JSON serializable, False otherwise.
    """
    try:
        json.dumps(obj, cls=NumpyEncoder)
        return True
    except (TypeError, OverflowError):
        return False


def save_clf(clf, filename, save_json=True, metadata=None):
    """
    Saves a scikit-learn classifier to a compressed pickle file, with an optional
    JSON sidecar containing parameters and training code metadata.

    Parameters
    ----------
    clf : object
        A scikit-learn classifier object to be saved.

    filename : str
        Full path (including filename) where the classifier will be saved. If the
        filename has no extension, '.pkl.gz' is appended. A JSON sidecar is saved
        to the same directory if `save_json` is True.

    save_json : bool, default=True
        Whether to save a JSON sidecar file containing classifier parameters and
        the training code context.

    metadata : dict, optional
        Additional metadata to include in the JSON sidecar file. If provided,
        must be a dictionary. Will be merged with classifier parameters and code.

    Returns
    -------
    clf_path : str
        Full path to the saved classifier file.

    Notes
    -----
    - The classifier is saved using `joblib.dump` with gzip compression.
    - The JSON sidecar includes:
        - Classifier hyperparameters from `get_params()`, with non-serializable
          objects converted to strings.
        - The source code line from which `save_clf` was called (using `inspect`).
        - Any additional user-provided metadata.
    - Filenames follow the pattern `<base>_clf.pkl.gz` and `<base>.json`, where
      `<base>` is derived from `filename` (without extension).
    - If the target directory does not exist, it is created automatically.
    """
    # Set default name if none is provided
    if filename is None:
        # Get the classifier's class name
        classifier_name = clf.__class__.__name__
        name = classifier_name.lower()
        base_fname = f"{name}_clf"
    else:
        if '.pkl' in filename:
            raise ValueError(f'{filename} contains pickle ending, please give name without file ending')
        base_fname = os.path.basename(filename).split('.')[0]

    assert not base_fname.startswith('.'), f'cannot save to hidden {base_fname=}'

    folder = os.path.dirname(filename)

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Full filenames for the classifier and JSON sidecar
    clf_fname = f"{base_fname}.pkl.gz"
    json_fname = f"{base_fname}.json"

    # Save the classifier using joblib with compression
    clf_path = os.path.join(folder, clf_fname)
    joblib.dump(clf, clf_path, compress=('gzip', 9))

    if not save_json:
        assert metadata is None
    elif save_json:

        # Get classifier parameters
        params = clf.get_params()
        params = {key:(val if is_json_serializable(val) else str(val)) for key, val in params.items()}

        # Retrieve the code where `save_clf` was called
        try:
            caller_frame = inspect.stack()[1]
            code_context = caller_frame.code_context
            code = ''.join(code_context).strip() if code_context else ''
        except:
            warnings.warn('ERROR RETRIEVING CODE')
            code = 'ERROR RETRIEVING CODE'

        # Create metadata dictionary
        if metadata is None:
            metadata = {}

        assert isinstance(metadata, dict)
        metadata |= {
            'classifier_parameters': params,
            'training_code': code
        }

        # Save metadata to JSON sidecar file
        json_path = os.path.join(folder, json_fname)
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    return clf_path

# @profile
def cross_validation_across_time(data_x, data_y, clf, add_null_data=False,
                                 n_jobs=-2, plot_confmat=False, title_add="",
                                 ex_per_fold=2, simulate=False, subj="",
                                 tmin=-0.1, tmax=0.5, sfreq=100,
                                 return_probas=False,
                                 verbose=True):
    """
    Perform cross-validation across time on the given dataset.

    Parameters
    ----------
    data_x : ndarray
        The input data array with shape (n_samples, n_features, n_timepoints).
    data_y : ndarray
        The target labels array with shape (n_samples,).
    clf : object
        The classifier object that implements the `fit` and `predict` methods.
    add_null_data : bool, optional
        If True, add null data to the training set. Default is False.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is -2.
    plot_confmat : bool, optional
        If True, plot the confusion matrix. Default is False.
    title_add : str, optional
        Additional title information for plots. Default is an empty string.
    verbose : bool, optional
        If True, print progress information. Default is True.
    ex_per_fold : int, optional
        The number of examples per fold. Default is 2.
    simulate : bool, optional
        If True, simulate data. Default is False.
    subj : str, optional
        Subject identifier. Default is an empty string.
    ms_per_point : int, optional
        Milliseconds per time point. Default is 10.
    return_preds : bool, optional
        If True, return predictions along with the DataFrame. Default is False.

    Returns
    -------
    df : DataFrame
        A DataFrame containing the cross-validation results.
    all_preds : ndarray, optional
        An array of predictions if `return_preds` is True.
    """
    # Ensure each class has the same number of examples
    assert (len(set(np.bincount(data_y)).difference(set([0]))) == 1), \
        "WARNING not each class has the same number of examples"
    # warnings.warn('RETURN THIS')
    # Set random seed based on subject ID for reproducibility
    np.random.seed(misc.string_to_seed(subj))

    # Get unique labels and create index tuples for cross-validation
    labels = np.unique(data_y)
    idxs_tuples = np.array([np.where(data_y == cond)[0] for cond in labels]).T
    idxs_tuples = [
        idxs_tuples[i: i + ex_per_fold].ravel()
        for i in range(0, len(idxs_tuples), ex_per_fold)
    ]

    # Determine the maximum time point
    time_max = data_x.shape[-1]  # 500 ms

    # Initialize progress bar and results DataFrame
    total = len(idxs_tuples)
    tqdm_loop = tqdm(total=total, desc=f"CV Fold {subj}", disable=not verbose)
    df = pd.DataFrame()

    # Initialize array to store all predictions
    all_probas = np.zeros([len(data_y), time_max, len(labels)])

    times = np.linspace(tmin*1000, tmax*1000, time_max).round()

    # Iterate over each fold
    for j, idxs in enumerate(idxs_tuples):
        # Split data into training and testing sets
        idxs_arr = np.zeros(data_x.shape[0], dtype=bool)
        idxs_arr[idxs] = True
        idxs_train = ~idxs_arr
        idxs_test = idxs_arr

        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]
        test_x = data_x[idxs_test]
        test_y = data_y[idxs_test]

        # Add null data if specified
        neg_x = np.hstack(train_x[:, :, 0:1].T).T if add_null_data else None

        # Train and predict in parallel across time points
        probas = Parallel(n_jobs=n_jobs)(
            delayed(train_predict)(
                train_x=train_x[:, :, start],
                train_y=train_y,
                test_x=test_x[:, :, start],
                neg_x=neg_x,
                clf=clf,
                proba=True
                # ova=ova,
            )
            for start in list(range(0, time_max))
        )
        probas = np.swapaxes(probas, 0, 1)

        # Store predictions and calculate accuracy
        all_probas[idxs_test] = probas

        preds = np.argmax(probas, -1)

        accuracy = (preds == test_y[:, None]).mean(axis=0)

        # Create a temporary DataFrame for the current fold
        df_temp = pd.DataFrame(
            {"timepoint": times,
             "fold": [j] * len(accuracy),
             "accuracy": accuracy,
             "subject": [subj] * len(accuracy),
            }
        )
        # Concatenate the temporary DataFrame with the main DataFrame
        df = pd.concat([df, df_temp], ignore_index=True)

        # Update progress bar
        tqdm_loop.update()

    # Close the progress bar
    tqdm_loop.close()

    # Return results
    return (df, all_probas) if return_probas else df



def decoding_heatmap_transfer(clf, data_x, data_y, test_x, test_y, range_t=None):
    """
    create a heatmap of decoding by varying training and testing time of
    two independent samples
    """
    heatmap = np.zeros([data_x.shape[-1], test_x.shape[-1]])
    for t_train in tqdm(range(data_x.shape[-1]), desc='train_predict'):
        clf.fit(data_x[:, :, t_train], data_y)
        for t_test in range(test_x.shape[-1]):
            acc = (clf.predict(test_x[:, :, t_test]) == test_y).mean()
            heatmap[t_train, t_test] = acc
    return heatmap


def decoding_heatmap_generalization(clf, data_x, data_y, ex_per_fold=4, n_jobs=8, range_t=None):
    """

    using cross validation, create a heatmap of decoding by varying training
    and testing times

    :param clf: classifier to use for creation of the heatmap
    :param data_x: data to train on
    :param data_y: data to test on
    :param ex_per_fold: DESCRIPTION, defaults to 4
    :param n_jobs: DESCRIPTION, defaults to 8
    :param range_t: DESCRIPTION, defaults to None
    :return: DESCRIPTION
    :rtype: np.array of

    """
    assert (
        len(set(np.bincount(data_y)).difference(set([0]))) == 1
    ), "WARNING not each class has the same number of examples"
    np.random.seed(0)
    labels = np.unique(data_y)
    idxs_tuples = np.array([np.where(data_y == cond)[0] for cond in labels]).T
    idxs_tuples = [
        idxs_tuples[i : i + ex_per_fold].ravel()
        for i in range(0, len(idxs_tuples), ex_per_fold)
    ]

    if range_t is None:
        range_t = np.arange(data_x.shape[-1])

    tqdm_total = len(idxs_tuples) * (len(range_t) ** 2)
    res = np.zeros([len(idxs_tuples), len(range_t), len(range_t)])

    for i, idxs in enumerate(idxs_tuples):
        idxs_train = ~np.in1d(range(data_x.shape[0]), idxs)
        idxs_test = np.in1d(range(data_x.shape[0]), idxs)
        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]
        test_x = data_x[idxs_test]
        test_y = data_y[idxs_test]
        params = list(itertools.product(range_t, range_t))
        tqdm_initial = i * len(range_t) ** 2
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_predict)(
                train_x[:, :, train_at],
                train_y,
                test_x[:, :, predict_at],
                clf=clf,
                # ova=False,
            )
            for train_at, predict_at in tqdm(
                params, total=tqdm_total, initial=tqdm_initial
            )
        )
        accs = np.mean(np.array(results) == test_y, -1)
        res[i, :, :] = accs.reshape([len(range_t), len(range_t)])
    return res.mean(0).squeeze()

def train_predict(train_x, train_y, test_x, clf, neg_x=None, proba=False):
    """
    Train a classifier with the given data and return predictions on test data.

    Parameters:
    train_x (array-like): The training input samples. Must be a 2D array.
    train_y (array-like): The target values (class labels) for the training input samples.
    test_x (array-like): The input samples for which predictions are to be made. Must be a 2D array.
    clf (object): The classifier object that implements the 'fit' and 'predict' methods.
                            If None, a default classifier should be provided outside this function.
    neg_x (array-like, optional): Additional negative class samples that contain no stimulation
                                  (e.g., a "null class"). Used for training if provided.
    proba (bool, optional): If True, the function returns class probabilities.
                            If False, it returns class labels. Default is False.

    Returns:
    array-like: The predicted class labels or probabilities for the test data, depending on the 'proba' parameter.

    Raises:
    AssertionError: If test_x is not a 2D array.
    """
    assert test_x.ndim == 2, "test data must be 2d"
    if neg_x is None:
        clf.fit(train_x, train_y)
    else:
        clf.fit(train_x, train_y, neg_x=neg_x)
    if proba:
        return clf.predict_proba(test_x)
    return clf.predict(test_x)


def get_mean_corrcoef(arr):
    """calculate corrcoeff for matrix and return mean off-diagonal correlation"""
    corrcoef = np.abs(np.corrcoef(arr))
    mean_corrcoef = np.nanmean(corrcoef[~np.eye(len(corrcoef), dtype=bool)])
    return mean_corrcoef


def get_channel_importances(data_x, data_y, n_folds=5,
                            n_trees=500, **kwargs):
    """returns the importance of features from a RandomForest cross-validation"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    assert data_x.ndim==2, 'must select timepoint before already, data must be 2d'
    rankings = []
    accs = []
    clf = RandomForestClassifier(n_trees, n_jobs=-1)
    cv = StratifiedKFold(n_folds, shuffle=True)

    for idxs_train, idx_test in cv.split(data_x, data_y):
        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]

        test_x = data_x[idx_test]
        test_y = data_y[idx_test]

        clf.fit(train_x, train_y)
        acc = np.mean(clf.predict(test_x)==test_y)
        ranking = clf.feature_importances_
        rankings.append(ranking)
        accs.append(acc)

    return accs, np.mean(rankings, 0)


def to_long_df(arr, columns=None, value_name='value', **col_labels):
    """Convert an N-D numpy array to a long-format DataFrame; include only labeled dims.

    Args:
        arr: N-D numpy array.
        columns: Sequence of dimension names; defaults to dim1..dimN.
        value_name: Name for the values column.
        **col_labels: For each dimension in `columns`, either
            - a 1-D sequence of labels (length == size of that axis), producing one column
              named as the dimension; or
            - a dict mapping {output_col_name -> 1-D sequence of labels} to produce
              multiple columns from the same axis (each sequence length must match axis size).

    Returns:
        DataFrame with columns [value_name, *labeled_columns], ordered by Fortran ('F') traversal.
        Dimensions not present in `col_labels` are omitted.
    """
    arr = np.asarray(arr)
    ndim = arr.ndim

    if columns is None:
        columns = [f'dim{i+1}' for i in range(ndim)]
    elif len(columns) != ndim:
        raise ValueError("len(columns) must match arr.ndim")

    # Validate kwargs names early
    unknown = set(col_labels).difference(columns)
    if unknown:
        raise KeyError(f"Unknown column(s) in col_labels: {sorted(unknown)}; valid names: {columns}")

    # Fortran-order linearization to match arr.ravel('F')
    n = arr.size
    lin = np.arange(n)
    coords = np.array(np.unravel_index(lin, arr.shape, order='F')).T  # (n, ndim)

    # Assemble output
    out_data = {value_name: arr.ravel('F')}
    out_cols = [value_name]
    used_colnames = set(out_cols)

    for ax, dim_name in enumerate(columns):
        if dim_name not in col_labels:
            continue

        spec = col_labels[dim_name]

        # Single sequence ? one column named after the dimension
        if not isinstance(spec, dict):
            labels = np.asarray(spec)
            if labels.ndim != 1 or labels.size != arr.shape[ax]:
                raise ValueError(f"Labels for '{dim_name}' must be 1-D of length {arr.shape[ax]}")
            out_name = dim_name
            if out_name in used_colnames:
                raise ValueError(f"Duplicate output column name: '{out_name}'")
            out_data[out_name] = labels[coords[:, ax]]
            out_cols.append(out_name)
            used_colnames.add(out_name)
            continue

        # Dict ? multiple output columns
        for out_name, labels in spec.items():
            labels = np.asarray(labels)
            if labels.ndim != 1 or labels.size != arr.shape[ax]:
                raise ValueError(
                    f"Labels for '{dim_name}.{out_name}' must be 1-D of length {arr.shape[ax]}"
                )
            if out_name in used_colnames:
                raise ValueError(f"Duplicate output column name: '{out_name}'")
            out_data[out_name] = labels[coords[:, ax]]
            out_cols.append(out_name)
            used_colnames.add(out_name)

    return pd.DataFrame(out_data, columns=out_cols)




class TimeEnsembleVoting(VotingClassifier):
    """
    TimeVotingClassifier trains one clone of a base estimator on each time slice
    of a 3D input (n_samples, n_features, n_times). It stores all these estimators
    in a VotingClassifier to leverage its predict/predict_proba functionality.

    Parameters
    ----------
    base_estimator : estimator object
        The classifier to clone and fit on each time point.

    voting : str, {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities.

    weights : array-like of shape (n_estimators,), default=None
        Sequence of weights (float values) to weight the occurrences of predicted class
        labels (hard voting) or class probabilities before averaging (soft voting).

    n_jobs : int, default=None
        Number of jobs to run in parallel for fit. None means 1 unless in
        a joblib.parallel_backend context. -1 means using all processors.

    flatten_transform : bool, default=True
        Parameter inherited from VotingClassifier. Ignored unless voting='soft'.
    """

    def __init__(
        self,
        base_estimator,
        voting='hard',
        weights=None,
        n_jobs=None,
        flatten_transform=True
    ):
        self.base_estimator = base_estimator
        super().__init__(
            estimators=[(f'{base_estimator}', self.base_estimator), ],
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform
        )

    def fit(self, X, y, *, sample_weight=None, **fit_params):
        """Get common fit operations."""
        # get all the imports that the votingclassifier also uses

        if len(X.shape) != 3:
            raise ValueError(
                "X should be of shape (n_samples, n_features, n_times). "
                f"Got {X.shape}."
            )

        n_samples, n_features, n_times = X.shape

        # part of VotingClassifier
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        names, clfs = self._validate_estimators()
        assert len(clfs)==1,\
            'only one base_estimator should be supplied to TimeEnsembleVoting'

        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError(
                "Number of `estimators` and weights must be equal; got"
                f" {len(self.weights)} weights, {len(self.estimators)} estimators"
            )

        # create copies of the clf
        clfs = clfs * n_times
        names = names * n_times

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch()
            for name in names:
                routed_params[name] = Bunch(fit={})
                if "sample_weight" in fit_params:
                    routed_params[name].fit["sample_weight"] = fit_params[
                        "sample_weight"
                    ]

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X[:,:,idx],
                transformed_y,
                fit_params=routed_params[name]["fit"],
                message_clsname="Voting",
                message=self._log_message(name, idx + 1, len(clfs)),
            )
            for idx, (name, clf) in enumerate(zip(names, clfs))
            if clf != "drop"
        )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        return self



class LogisticRegressionOvaNegX(LogisticRegression):
    """one vs all logistic regression classifier including negative examples.

    Under the hood, one separate LogisticRegression is trained per class.
    The LogReg is trained using positive examples (inclass) and negative
    examples (outclass + nullclass).
    """

    def __init__(
        self,
        base_clf=None,
        penalty="l1",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        neg_x_ratio=1.0,
    ):
        self.base_clf = None if base_clf is None else base_clf  # just for __repr__

        if base_clf is None:
            base_clf = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=max_iter,
                # multi_class="ovr",
            )
        assert is_classifier(
            base_clf
        ), f"Must supply classifier, but supplied {base_clf}"
        self.base_clf_ = base_clf
        self.neg_x_ratio = neg_x_ratio
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        # self.n_pca = n_pca
        LogisticRegression.__init__(
            self,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            # multi_class="ovr",
        )

    def fit(self, X, y, neg_x=None, neg_x_ratio=None):
        # if self.n_pca is not None:
        #     self.pca = PCA(self.n_pca)
        #     X = self.pca.fit_transform(X)
        #     neg_x = self.pca.transform(neg_x)
        self.classes_ = np.unique(y)
        neg_x_ratio = self.neg_x_ratio if neg_x_ratio is None else neg_x_ratio
        models = []
        intercepts = []
        coefs = []

        for class_ in self.classes_:
            clf = clone(self.base_clf_)
            idx_class = y == class_
            true_x = X[idx_class]
            false_x = X[~idx_class]

            if neg_x is not None:
                n_null = int(len(X) * neg_x_ratio)
                replace = len(neg_x) < n_null
                idx_neg = np.random.choice(len(neg_x), size=n_null, replace=replace)
                false_x = np.vstack([false_x, neg_x[idx_neg]])

            data_x = np.vstack([true_x, false_x])
            data_y = np.hstack([np.ones(len(true_x)), np.zeros(len(false_x))])

            clf.fit(data_x, data_y)
            models.append(clf)
            intercepts.append(clf.intercept_)
            coefs.append(clf.coef_)

        self.models = models
        self.intercept_ = np.squeeze(intercepts)
        self.coef_ = np.squeeze(coefs)

        return self

    def predict_proba(self, X):
        # if self.n_pca is not None:
        #     X = self.pca.transform(X)
        proba = []
        for clf in self.models:
            p = clf.predict_proba(X)[:, 1]
            proba.append(p)
        return np.array(proba).T

#%% main
import stimer
if __name__=='__main__':
    data_x = np.random.rand(100, 306, 101)
    data_y = np.repeat(np.arange(10), 10)
    clf_base = LogisticRegressionOvaNegX(C=4.8)

    # cross_validation_across_time(data_x, data_y, clf)
    clf = TimeEnsembleVoting(clf_base, n_jobs=-1)
    clf.fit(data_x, data_y)
