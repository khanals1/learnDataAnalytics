import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
# dummy encode the genre
movies = movies.join(movies.genres.str.get_dummies("|"))
cos_sim = cosine_similarity(movies.iloc[:, 3:])
toystory_top5 = np.argsort(cos_sim[0])[-5:][::-1]

ratings = pd.read_csv("ratings.csv")

mean_rating = ratings['rating'].mean()  # compute mean rating

pref_matrix = ratings[['userId', 'movieId', 'rating']].pivot(index='userId', columns='movieId', values='rating')

pref_matrix = pref_matrix - mean_rating  # adjust by overall mean

item_mean_rating = pref_matrix.mean(axis=0)
pref_matrix = pref_matrix - item_mean_rating  # adjust by item mean

user_mean_rating = pref_matrix.mean(axis=1)
pref_matrix = pref_matrix - user_mean_rating

pref_matrix.fillna(0) + user_mean_rating + item_mean_rating + mean_rating

mat = pref_matrix.values
k = 0  # use the first user
np.nansum((mat - mat[k, :]) ** 2, axis=1).reshape(-1, 1)

np.nansum((mat - mat[0, :]) ** 2, axis=1)[1:].argmin()  # returns 11
# check it:
np.nansum(mat[12] - mat[0])  # returns 0.0

np.where(~np.isnan(mat[12]) & np.isnan(mat[0]) == True)
# returns (array([304, 596]),)

print(mat[12][[304, 596]])
# returns array([-2.13265214, -0.89476547])
