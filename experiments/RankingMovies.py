import sys
sys.path.insert(0, '../lib')  # noqa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn.metrics import *

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import csv
# from lasagne import layers
# from lasagne.updates import nesterov_momentum
# from lasagne.nonlinearities import softmax
#from nolearn.lasagne import NeuralNet
from greedy_order import *

genres_data = pd.read_csv(
    'movielens-dataset/u.genre',
    sep = '|',
    encoding = "ISO-8859-1",
    header = None,
    names=['name', 'id']
)
movie_data_columns = np.append(
    ['movie_id', 'title', 'release_date', 'video_release_date', 'url'],
    genres_data['name'].values
)

movie_data = pd.read_csv(
    'movielens-dataset/u.item',
    sep = '|',
    encoding = "ISO-8859-1",
    header = None,
    names = movie_data_columns,
    index_col = 'movie_id'
)
selected_columns = np.append(['title', 'release_date'], genres_data['name'].values)
movie_data = movie_data[selected_columns]
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])



ratings_data = pd.read_csv(
    'movielens-dataset/u.data',
    sep = '\t',
    encoding = "ISO-8859-1",
    header = None,
    names=['user_id', 'movie_id', 'rating', 'timestamp']
)


movie_data['ratings_average'] = ratings_data.groupby(['movie_id'])['rating'].mean()
movie_data['ratings_count'] = ratings_data.groupby(['movie_id'])['rating'].count()
print(movie_data.head())