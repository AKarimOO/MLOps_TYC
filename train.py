import sqlite3

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import common

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X = data_train.drop(columns=['trip_duration'])
    y = data_train['trip_duration']
    return X, y

def fit_baseline(y_train):
    y_baseline = y_train.mean()
    print(f'Baseline prediction: {y_baseline:.2f} (transformed)')
    print(f'Baseline prediction: {np.expm1(y_baseline):.0f} (seconds)')
    print(f'RMSLE on train data: {mean_squared_error([y_baseline] * len(y_train), y_train, squared=False):.3f}')

def fit_model1(X_train, y_train):
    X_train = common.step1_add_features(X_train)


    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), common.cat_features),
        ('scaling', StandardScaler(), common.num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(X_train[common.train_features], y_train)
    y_pred_train = model.predict(X_train[common.train_features])
    print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))
    return model








if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    y_train = common.transform_target(y_train)
    X_train = common.step1_add_features(X_train)


    print('--'*60)
    print('--'*60)

    fit_baseline(y_train)

    print('--' * 60)
    print('------------------MODEL1-------------')
    print('--' * 60)

    model1 = fit_model1(X_train, y_train)
    common.persist_model(model1, common.MODEL_PATH)


