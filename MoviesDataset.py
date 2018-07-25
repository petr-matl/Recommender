import numpy as np
import pandas as pd

class MoviesDataset:
    dataset = ''
    def __init__(self, dataset):
        self.dataset = dataset

    def getData(self, algorithm):
        if self.dataset == 'test':
            y = np.array([[5,5,0,0],[5,-1,-1,0],[-1,4,0,-1],[0,0,5,4],[0,0,5,-1]])
            r = y >= 0
            if algorithm == 'content-based':
                X = np.array([[1,1,0],[1,1,0],[1,1,0],[1,0,1],[1,0,1]])
                theta = np.random.randn(y.shape[1], X.shape[1])
            elif algorithm == 'colaborative':
                X = np.random.randn(y.shape[0], 2)
                theta = np.random.randn(y.shape[1], 2)
            #r = r.astype(int)
            return y, r, X, theta, y.shape[0], y.shape[1], X.shape[1]
        elif self.dataset == 'real':
            # detail information about movies
            genre_columns = ['Title', 'Genre id']
            genre_names = pd.read_csv('ml-100k/u.genre', sep='|', names=genre_columns, encoding='latin-1')

            movie_columns = ['Movie id', 'Title', 'Release date', 'Video release date', 'Imdb url']
            movie_columns.extend(genre_names['Title'])

            movie_features = pd.read_csv('ml-100k/u.item', sep='|', names=movie_columns, encoding='latin-1')
            movie_names = movie_features[['Movie id', 'Title']]
            movie_features.drop(['Title', 'Release date', 'Video release date', 'Imdb url', 'unknown'], axis=1, inplace=True)

            # detail information about users
            user_columns = ['User id', 'Age', 'Gender', 'Occupation', 'Zip code']
            user_features = pd.read_csv('ml-100k/u.user', sep='|', names=user_columns, encoding='latin-1')
            gender = pd.get_dummies(user_features['Gender'], drop_first=True)
            gender.rename(columns = {"M": "Male"}, inplace=True)
            user_features = pd.concat([user_features, gender], axis=1)
            user_features.drop(['Age', 'Occupation', 'Zip code', 'Gender'], axis=1, inplace=True)

            # information about users/movies interaction
            rating_columns = ['User id','Movie id','Rating','Timestamp']
            ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=rating_columns, encoding='latin-1')
            ratings.drop(['Timestamp'], axis=1, inplace=True)

            y = ratings.pivot_table(values='Rating', index=['Movie id'], columns=['User id']).fillna(0).values
            r = y > 0
            #r = ~np.isnan(y)
            if algorithm == 'content-based':
                X = movie_features.drop(['Movie id'], axis=1).values
                theta = np.random.randn(y.shape[1], X.shape[1])
            elif algorithm == 'colaborative':
                X = np.random.randn(y.shape[0], 10)
                theta = np.random.randn(y.shape[1], 10)
            return y, r, X, theta, y.shape[0], y.shape[1], X.shape[1]