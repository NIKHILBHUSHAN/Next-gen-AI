                              #1.Data Preprocessing:

import pandas as pd

# Load the data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge the ratings with movie titles
data = pd.merge(ratings, movies, on='movieId')

# Pivot the data to create a user-item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')



                           #2.User-based Collaborative Filtering:

from sklearn.metrics.pairwise import cosine_similarity

# Fill missing values with 0 (as unrated)
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Calculate user-user similarity matrix
user_similarity = cosine_similarity(user_movie_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to recommend movies for a user
def recommend_movies(user_id, n_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]  # Exclude the user themselves
    recommended_movies = []
    
    for similar_user in similar_users:
        similar_user_movies = user_movie_matrix.loc[similar_user].dropna().index
        recommended_movies.extend(similar_user_movies)
        if len(recommended_movies) >= n_recommendations:
            break
    
    return recommended_movies[:n_recommendations]

# Get top 5 movie recommendations for user with ID 1
print(recommend_movies(1, 5))



                            #3.Matrix Factorization (SVD):

from sklearn.decomposition import TruncatedSVD

# Use TruncatedSVD for matrix factorization
svd = TruncatedSVD(n_components=20)
user_movie_matrix_filled_svd = svd.fit_transform(user_movie_matrix_filled)

# Convert back to a user-item matrix for predictions
predicted_ratings = svd.inverse_transform(user_movie_matrix_filled_svd)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# Get recommendations based on highest predicted ratings for a user
def recommend_by_svd(user_id, n_recommendations=5):
    user_predictions = predicted_ratings_df.loc[user_id].sort_values(ascending=False)
    return user_predictions.index[:n_recommendations]

# Recommend movies for user with ID 1
print(recommend_by_svd(1, 5))


                         #4.Evaluation:
from sklearn.metrics import mean_squared_error

# Get RMSE on known ratings
known_ratings = user_movie_matrix_filled.values.flatten()
predicted_ratings_flat = predicted_ratings_df.values.flatten()
rmse = mean_squared_error(known_ratings, predicted_ratings_flat, squared=False)

print(f'RMSE: {rmse}')

