import sqlite3
import os

import pandas as pd
from sklearn.model_selection import train_test_split
import common

# use this value where it is possible to indicate the random state
RANDOM_STATE = 42


data = pd.read_csv('data/New_York_City_Taxi_Trip_Duration.csv')

# check if there are any columns containing unique values for each row. If so, drop them.
data = data.drop(columns=['id'])

# dropoff_datetime variable is added only to train data and thus cannot be used by the predictive model. Drop this feature.
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['pickup_date'] = data['pickup_datetime'].dt.date


data_train,data_test= train_test_split(data, test_size=0.3, random_state=RANDOM_STATE)




db_dir = os.path.dirname(common.DB_PATH)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

print(f"Saving train and test data to a database: {common.DB_PATH}")
with sqlite3.connect(common.DB_PATH) as con:
    # cur = con.cursor()
    # cur.execute("DROP TABLE IF EXISTS train")
    # cur.execute("DROP TABLE IF EXISTS test")
    data.to_sql(name='data', con=con, if_exists="replace")
    data_train.to_sql(name='train', con=con, if_exists="replace")
    data_test.to_sql(name='test', con=con, if_exists="replace")
