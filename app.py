import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import requests

# Load the pre-trained model and datasets
model = load_model('recommender_model.h5')
movies = pd.read_pickle('movies.pkl')
newratedmovies = pd.read_pickle('newratedmovies.pkl')


# Function to fetch movie poster URL from the posters DataFrame
def fetch_poster(movie_id):
    # Extract the movie with the given ID
    movie = movies[movies['movie_id'] == movie_id]
    if movie.empty:
        return None

    # Extract the year and title separately
    full_title = movie['title'].values[0]
    year = full_title[-5:-1]  # Extract the year within the last parenthesis
    title = full_title[:-6]  # Extract the title part

    # Check if the title contains a comma followed by 'The', 'A', or 'An'
    if ',' in title:
        parts = title.split(', ')
        print(parts)
        if len(parts) == 2 and parts[1] in ['The ', 'A ', 'An ']:
            #remove white space from parts[1]
            parts[1] = parts[1].strip()
            title = f"{parts[1]} {parts[0]}"  # Rearrange as "The Shining" for "Shining, The"
    # Search for the movie using the OMDb API
    url = f'http://www.omdbapi.com/?apikey=6787511b&t={title}&y={year}'
    response = requests.get(url)
    data = response.json()
    
    

    # Get the poster URL from the JSON response
    if "Poster" in data and data["Poster"] != 'N/A':
        return data["Poster"]
    else:
        return None

    

# Functions for getting movie ID and user ID
def get_movie_id(movie_name):
    movie = movies[movies['title'] == movie_name]
    return movie['movie_id'].values[0] if not movie.empty else None

def get_user(movie_id):
    user = newratedmovies[newratedmovies['movie_id'] == movie_id]
    return user['user_id'].values[0] if not user.empty else None

def predict_rating(user_id, movie_id):
    user = np.array([user_id])
    movie = np.array([movie_id])
    genre = np.array([newratedmovies[newratedmovies['movie_id'] == movie_id]['genres'].values[0]])
    prediction = model.predict([user, movie, genre])
    return prediction[0][0]

# Streamlit UI setup
st.header("Movie Recommender System")

# Dropdown for selecting a movie
movies_list = movies['title'].values
selected_movie = st.selectbox("Select a movie from the dropdown", movies_list)

# Show recommendations when the button is clicked
if st.button("Show Recommendations"):
    movie_id = get_movie_id(selected_movie)
    if movie_id:
        user_id = get_user(movie_id)
        if user_id:
            # Get user-rated movies and make predictions
            user_ratings = newratedmovies[newratedmovies['user_id'] == user_id][['user_id', 'movie_id', 'rating']]
            user_ratings['predicted_rating'] = user_ratings.apply(lambda x: predict_rating(user_id, x['movie_id']), axis=1)
            user_ratings = user_ratings.sort_values(by='predicted_rating', ascending=False).merge(movies, on='movie_id', how='inner')
            
            # Get the genres of the specified movie
            movie_genres = movies[movies['movie_id'] == movie_id]['genres'].values[0].split('|')  # Assuming genres are "|" separated

            # Retrieve user ratings and predict ratings for rated movies
            user_ratings = newratedmovies[newratedmovies['user_id'] == user_id][['user_id', 'movie_id', 'rating']]
            user_ratings['predicted_rating'] = user_ratings.apply(lambda x: predict_rating(user_id, x['movie_id']), axis=1)
            user_ratings = user_ratings.sort_values(by='rating', ascending=False).merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)


            # Get movie recommendations
            recommended_movies = newratedmovies[newratedmovies['movie_id'].isin(user_ratings['movie_id'])]['movie_id'].unique()
            recommendations = pd.DataFrame(recommended_movies, columns=['movie_id'])
            recommendations['predictions'] = recommendations.apply(lambda x: predict_rating(user_id, x['movie_id']), axis=1)

            # # Filter recommendations by genre similarity
            recommendations = recommendations.merge(movies[['movie_id', 'genres']], on='movie_id', how='inner')
            recommendations['genre_match'] = recommendations['genres'].apply(lambda g: len(set(g.split('|')) & set(movie_genres)))

            # # Sort by genre match (prioritized) and then by prediction score
            recommendations = recommendations.sort_values(by=['genre_match', 'predictions'], ascending=[False, False])

            # # Display the top 10 recommendations
            top_recommendations = recommendations.merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(5)

            # Display top 5 recommended movies
            cols = st.columns(5)
            for i, movie in enumerate(top_recommendations.itertuples(), start=1):
                with cols[i - 1]:
                    st.text(movie.title)
                    poster = fetch_poster(movie.movie_id)
                    if poster:
                        st.image(poster)
                    else:
                        st.text("Poster not available")
        else:
            st.error("No user found who rated the selected movie.")
    else:
        st.error("Movie not found in the dataset.")
