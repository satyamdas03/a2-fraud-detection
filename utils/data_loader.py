import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def download_dataset(path="creditcard.csv"):
    """Download the ULB creditcard.csv if it is not already present."""
    if os.path.exists(path):
        return path
    urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_data(path="creditcard.csv"):
    """Load the creditcard dataset from disk or URL."""
    download_dataset(path)
    df = pd.read_csv(path)
    return df


def temporal_split(df, train_frac=0.6, val_frac=0.2):
    """Split by Time to avoid future data leaking into training."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def prepare_features(df, scaler=None, fit_scaler=True):
    """Drop Time from features, scale Amount, keep V1-V28."""
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].values

    feature_cols = [c for c in X.columns if c != "Time"]
    X = X[feature_cols]

    if "Amount" in X.columns:
        if fit_scaler:
            scaler = StandardScaler()
            X["Amount"] = scaler.fit_transform(X[["Amount"]])
        else:
            X["Amount"] = scaler.transform(X[["Amount"]])

    return X, y, scaler


def get_splits(path="creditcard.csv"):
    """Full pipeline: load, split, scale."""
    df = load_data(path)
    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train, scaler = prepare_features(train_df, fit_scaler=True)
    X_val, y_val, _ = prepare_features(val_df, scaler=scaler, fit_scaler=False)
    X_test, y_test, _ = prepare_features(test_df, scaler=scaler, fit_scaler=False)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
