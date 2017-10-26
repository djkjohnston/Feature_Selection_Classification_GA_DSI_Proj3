import random
from time import sleep

import pandas as pd
import numpy as np
import psycopg2 as pg2
from psycopg2.extras import RealDictCursor

import matplotlib.pyplot as plt
import seaborn as sns

# Get 3 sets of UCI Madelon data with 440 rows each.
# set as function to keep environment tidy

def load_uci():
    # The data to load. 
    data_csv = "data/manelon_complete_training.csv"
    target_csv = "data/manelon_complete_labels.csv"

    # Count the lines
    num_lines = sum(1 for l in open(target_csv))

    # Sample size - in this case ~10%
    size = 440

    # The row indices to skip - make sure 0 is not included to keep the header!
    random.seed(42) #for reproducable results

    skip_idx1 = random.sample(range(1, num_lines+1), num_lines - size)
    skip_idx2 = random.sample(range(1, num_lines+1), num_lines - size)
    skip_idx3 = random.sample(range(1, num_lines+1), num_lines - size)

    # Read the data
    Xuci_1 = pd.read_csv(data_csv, skiprows=skip_idx1)
    Xuci_2 = pd.read_csv(data_csv, skiprows=skip_idx2)
    Xuci_3 = pd.read_csv(data_csv, skiprows=skip_idx3)
    yuci_1 = pd.Series(pd.read_csv(target_csv, skiprows=skip_idx1)['target'])
    yuci_2 = pd.Series(pd.read_csv(target_csv, skiprows=skip_idx2)['target'])
    yuci_3 = pd.Series(pd.read_csv(target_csv, skiprows=skip_idx3)['target'])
    yuci_1 = yuci_1.replace(-1, 0)
    yuci_2 = yuci_2.replace(-1, 0)
    yuci_3 = yuci_3.replace(-1, 0)

    return Xuci_1, Xuci_2, Xuci_3, yuci_1, yuci_2, yuci_3

Xuci_1, Xuci_2, Xuci_3, yuci_1, yuci_2, yuci_3 = load_uci()


# For loading in data from postgres database. Setup as function to make sure that close() is run.
def madelon_db_sql(sql):
    con = pg2.connect(host='34.211.227.227',
                  dbname='postgres',
                  user='postgres')
    cur = con.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql)
    results = cur.fetchall()
    con.close()
    return results

# Downsampled to 1% of data for now. 
# Do feature selection on this smaller set of data and then select the full 10% with the selected features
def load_db():
    sql_1 = "SELECT * FROM madelon TABLESAMPLE BERNOULLI (1) REPEATABLE (42*1)"
    sql_2 = "SELECT * FROM madelon TABLESAMPLE BERNOULLI (1) REPEATABLE (42*2)"
    sql_3 = "SELECT * FROM madelon TABLESAMPLE BERNOULLI (1) REPEATABLE (42*3)"

    df_db_1 = madelon_db_sql(sql_1)
    sleep(1)
    df_db_2 = madelon_db_sql(sql_2)
    sleep(1)
    df_db_3 = madelon_db_sql(sql_3)
    df_db_1 = pd.DataFrame(df_db_1)
    df_db_2 = pd.DataFrame(df_db_2)
    df_db_3 = pd.DataFrame(df_db_3)
    
    Xdb_1 = df_db_1.drop(['_id','target'], axis=1)
    Xdb_2 = df_db_2.drop(['_id','target'], axis=1)
    Xdb_3 = df_db_3.drop(['_id','target'], axis=1)
    ydb_1 = df_db_1['target']
    ydb_2 = df_db_2['target']
    ydb_3 = df_db_3['target']
    
    return Xdb_1, Xdb_2, Xdb_3, ydb_1, ydb_2, ydb_3

Xdb_1, Xdb_2, Xdb_3, ydb_1, ydb_2, ydb_3 = load_db()