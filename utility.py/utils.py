import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any


def target_windower(dataset:pd.DataFrame, group_column:str, target_column:str, window_size:int, stride:Any = None) -> pd.Series:
    """
       A function for aligning class labels to the windows in the windowed feature matrix

    Args:
        dataset (DataFrame): dataset containing the target variable
        group_column (str): the column to be used for windowing
        target_column (str): column of the target variable
        window_size (int): number of rows in each window
        stride (Any, optional): determines whether the created windows are overlapping or not. Defaults to None.

    Returns:
        pd.Series: labels for each window in the feature matrix
    """
    stride = window_size if stride is None else stride
    window_labels = []

    for _, group_data in dataset.groupby(group_column):
        group_labels = group_data[target_column].reset_index(drop=True)

        for start in range(0, len(group_labels) - window_size + 1, stride):
            window_y = group_labels.iloc[start:start + window_size]
            window_labels.append(window_y.iloc[0])

    return pd.Series(window_labels).reset_index(drop=True)


class WindowTransformer(BaseEstimator, TransformerMixin):

    """
        A transformer that creates non-overlapping windows from time-series data.

    Args:
        window_size (int): Number of rows per window.
        stride (int, optional): Step between windows. Defaults to window_size for non-overlapping windows.
    """

    def __init__(self, window_size, group_column=None, stride=None, ):
        self.window_size = window_size
        self.stride = window_size if stride is None else stride
        self.group_column = group_column

    def fit(self, X, y=None):
        return self

    def _window_block(self, block) -> list[dict]:
        window_rows = []
        feature_cols = block.columns

        for start in range(0, len(block) - self.window_size + 1, self.stride):
            window = block.iloc[start:start + self.window_size]

            row = {}
            for t in range(self.window_size):
                for col in feature_cols:
                    row[f"{col}_t{t}"] = window.iloc[t][col]

            window_rows.append(row)

        return window_rows

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
            Transform the input data into one row per window.

        Args:
            X (DataFrame): Input data (feature matrix) to be transformed into windows.

        Returns:
            DataFrame: windowed data, where each row is a flattened window with preserved names,
            e.g. F3_t0, F3_t1, ..., Cz_t0, Cz_t1, ...
        """
        if self.group_column is None:
            return pd.DataFrame(self._window_block(X))

        all_rows = []
        feature_cols = [col for col in X.columns if col != self.group_column]

        for _, group_data in X.groupby(self.group_column, sort=False):
            block = group_data[feature_cols].reset_index(drop=True)
            all_rows.extend(self._window_block(block))

        return pd.DataFrame(all_rows)


class GroupedDataSplitter:
    """
    Split a dataset by whole groups so train and test contain disjoint subjects.

    Args:
        group_column (str): Column containing the subject or group id.
        train_ratio (float): Fraction of groups to place in the train set.
        shuffle (bool): Whether to shuffle group ids before splitting.
        random_state (int | None): Seed used when shuffling.
    """

    def __init__(self, group_column:str, train_ratio:int, shuffle:bool = True, random_state:Any = None):
        self.group_column = group_column
        self.train_ratio = train_ratio
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, dataset: pd.DataFrame) -> tuple:
        """
            Transform the input data into training and testing sets based on groups.

        Args:
            dataset (DataFrame): Input data to be split into training and testing sets.

        Returns:
            tuple: A tuple containing the training and testing data (train_data, test_data).
        """
        group_ids = dataset[self.group_column].drop_duplicates().to_numpy()

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(group_ids)

        split_index = int(len(group_ids) * self.train_ratio)
        train_group_ids = set(group_ids[:split_index])
        test_group_ids = set(group_ids[split_index:])

        train_data = dataset[dataset[self.group_column].isin(train_group_ids)].copy()
        test_data = dataset[dataset[self.group_column].isin(test_group_ids)].copy()

        return train_data, test_data


