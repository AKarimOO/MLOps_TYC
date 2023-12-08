import pickle
import os
import sqlite3

import numpy as np
import pandas as pd

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))

# # Doing the same with a YAML configuration file
# import yaml
#
# with open("config.yml", "r") as f:
#     config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
#     DB_PATH = str(config_yaml['paths']['db_path'])
#     MODEL_PATH = str(config_yaml['paths']["model_path"])
#     RANDOM_STATE = int(config_yaml["ml"]["random_state"])

# SQLite requires the absolute path
# DB_PATH = os.path.abspath(DB_PATH)
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

def preprocess_data(X):
    print(f"Preprocessing data")
    return X

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model

def load_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data = pd.read_sql('SELECT * FROM data', con)
    con.close()
    return data

data = load_data(DB_PATH)
X = data.drop(columns=['trip_duration'])
y = data['trip_duration']
def transform_target(y):
    return np.log1p(y).rename('log_' + y.name)

data = load_data(DB_PATH)
X=data.drop(columns=['trip_duration'])
y = data['trip_duration']


df_abnormal_dates = X.groupby('pickup_date').size()
abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]


num_features = ['abnormal_period', 'hour']
cat_features = ['weekday', 'month']
train_features = num_features + cat_features


num_features2 = ['log_distance_haversine', 'hour',
                    'abnormal_period', 'is_high_traffic_trip', 'is_high_speed_trip',
                    'is_rare_pickup_point', 'is_rare_dropoff_point']
cat_features2 = ['weekday', 'month']

train_features2 = num_features2 + cat_features2


def step1_add_features(X):
  res = X.copy()
  res['pickup_datetime'] = pd.to_datetime(res['pickup_datetime'])
  res['weekday'] = res['pickup_datetime'].dt.weekday
  res['month'] = res['pickup_datetime'].dt.month
  res['hour'] = res['pickup_datetime'].dt.hour
  res['abnormal_period'] = res['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
  return res

