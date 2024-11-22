# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:59:16 2024

@author: Simon Kern
"""
import numpy as np
import pandas as pd
from. import misc
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed

# @memory.cache(ignore=["n_jobs", "plot_confmat", "title_add", "verbose"])


def cross_validation_across_time(data_x, data_y, clf, add_null_data=False,
                                 n_jobs=-2, plot_confmat=False, title_add="",
                                 ex_per_fold=2, simulate=False, subj="",
                                 tmin=-100, tmax=500,
                                 ms_per_point=10, return_preds=False,
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
    # assert (len(set(np.bincount(data_y)).difference(set([0]))) == 1), \
    #     "WARNING not each class has the same number of examples"
    warnings.warn('RETURN THIS')
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
    all_preds = np.zeros([time_max, len(data_y)], dtype=int)

    # Iterate over each fold
    for j, idxs in enumerate(idxs_tuples):
        # Split data into training and testing sets
        idxs_train = ~np.in1d(range(data_x.shape[0]), idxs)
        idxs_test = np.in1d(range(data_x.shape[0]), idxs)
        train_x = data_x[idxs_train]
        train_y = data_y[idxs_train]
        test_x = data_x[idxs_test]
        test_y = data_y[idxs_test]

        # Add null data if specified
        neg_x = np.hstack(train_x[:, :, 0:1].T).T if add_null_data else None

        # Train and predict in parallel across time points
        preds = Parallel(n_jobs=n_jobs)(
            delayed(train_predict)(
                train_x=train_x[:, :, start],
                train_y=train_y,
                test_x=test_x[:, :, start],
                neg_x=neg_x,
                clf=clf,
                # ova=ova,
            )
            for start in list(range(0, time_max))
        )

        # Store predictions and calculate accuracy
        all_preds[:, idxs_test] = np.array(preds)
        acc_hit_miss = np.array([pred == test_y for pred in preds]).mean(-1)

        # Create a temporary DataFrame for the current fold
        times = np.linspace(tmin*1000, tmax*1000, len(preds)).round()
        df_temp = pd.DataFrame(
            {
                "timepoint": times,
                "fold": [j] * len(acc_hit_miss),
                "accuracy": acc_hit_miss,
                "preds": preds,
                "subject": [subj] * len(acc_hit_miss),
            }
        )
        # Concatenate the temporary DataFrame with the main DataFrame
        df = pd.concat([df, df_temp], ignore_index=True)

        # Update progress bar
        tqdm_loop.update()

    # Close the progress bar
    tqdm_loop.close()

    # Return results
    return (df, all_preds) if return_preds else df


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
