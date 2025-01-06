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
from tqdm import tqdm
import warnings
import inspect
import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator, clone, is_classifier

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
    JSON sidecar containing parameters and training code.

    Parameters:
    -----------
    clf : object
        Scikit-learn classifier object to save.

    name : str, optional
        Identifier for the saved classifier. If `None`, the classifier's class
        name is used. This value is incorporated into a BIDS-compatible filename.

    folder : str, optional
        Directory where the classifier and associated files will be saved.
        Defaults to the current working directory.

    save_json : bool, default True
        If `True`, saves the classifier's parameters and training code in a
        sidecar JSON file.

    Notes:
    ------
    - The classifier is saved using compressed pickle format (`.pkl.gz`).
      If the provided filename does not end with a compression extension
      (e.g., `.zip`, `.gz`), `.pkl.gz` is appended.
    - The JSON sidecar file contains the classifier's parameters and the code
      context where  `save_clf` was called, retrieved using the `inspect` module.
    - The filenames are constructed to be BIDS-compatible, following the pattern:
      `sub-group_desc-<name>_clf.<extension>`.
    """

    # Set default name if none is provided
    if filename is None:
        # Get the classifier's class name
        classifier_name = clf.__class__.__name__
        name = classifier_name.lower()
        base_fname = f"{name}_clf"
    else:
        # Remove any existing file extension
        base_fname = Path(filename).stem

    folder = os.path.dirname(filename)

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    base_fname = f'{folder}/{base_fname}'
    # Full filenames for the classifier and JSON sidecar
    clf_fname = f"{base_fname}.pkl.gz"
    json_fname = f"{base_fname}.json"

    # Save the classifier using joblib with compression
    clf_path = os.path.join(folder, clf_fname)
    joblib.dump(clf, clf_path, compress=True)

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
                                 tmin=-100, tmax=500,
                                 ms_per_point=10, return_probas=False,
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
    times = np.linspace(tmin * 1000, tmax * 1000, time_max).round()

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


def train_predict(train_x, train_y, test_x, clf=None, neg_x=None, proba=False):
    """
    Train a classifier with the given data and return predictions on test data.

    Parameters:
    train_x (array-like): The training input samples. Must be a 2D array.
    train_y (array-like): The target values (class labels) for the training input samples.
    test_x (array-like): The input samples for which predictions are to be made. Must be a 2D array.
    clf (object, optional): The classifier object that implements the 'fit' and 'predict' methods.
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
    pred = clf.predict_proba(test_x) if proba else clf.predict(test_x)
    return pred


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
                multi_class="ovr",
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
            multi_class="ovr",
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


if __name__=='__main__':
    data_x = np.random.rand(100, 306, 101)
    data_y = np.repeat(np.arange(10), 10)
    clf = LogisticRegressionOvaNegX(C=4.8)

    cross_validation_across_time(data_x, data_y, clf)
