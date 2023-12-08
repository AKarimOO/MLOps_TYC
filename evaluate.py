import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import common

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']
    return X, y


def evaluate_baseline(y_test):
    y_test= common.transform_target(y_test)
    y_baseline = y_test.mean()
    print(f'Baseline prediction: {y_baseline:.2f} (transformed)')
    print(f'Baseline prediction: {np.expm1(y_baseline):.0f} (seconds)')
    print(f'RMSLE on test data: {mean_squared_error([y_baseline] * len(y_test), y_test, squared=False):.3f}')

def evaluate_model1(model1, X_test, y_test):
    print(f"Evaluating the model")
    y_pred = model1.predict(X_test[common.train_features])
    rems = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rems,r2

if __name__ == "__main__":
    X_test, y_test = load_test_data(common.DB_PATH)
    y_test = common.transform_target(y_test)
    X_test = common.step1_add_features(X_test)

    print('-' * 60)
    print('-'*60)

    evaluate_baseline(y_test)

    print('--' * 60)
    print('------------------MODEL1-------------')
    print('--' * 60)

    X_test = common.preprocess_data(X_test)
    model = common.load_model(common.MODEL_PATH)
    rems,r2 = evaluate_model1(model, X_test, y_test)
    print("Test RMSE = %.4f" % rems)
    print("Test R2 = %.4f" % r2)








