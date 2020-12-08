"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st



# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    #movies['keyWords'] = movies['genres'].str.replace('|', ' ')

    #Remove punctuations
    #movies['title'] = movies['title'].str.replace('[^\w\s]','')
    
    movies['title'] = movies['title'].apply(lambda x: x.strip())
    #movies['title'] = movies['title'].str.replace('[^\w\s]','')
    movies.movieId = movies.movieId.astype('int32')
    ratings.rating=ratings.rating.astype('int32')
    movies['keywords'] = movies['genres'].str.replace('|', ' ')
    movies['keywords'] = movies['keywords'].apply(lambda x: x.lower())
    movies['keywords'] = movies['keywords'].apply(lambda x: x.strip())
    #movies['genres'] = movies.genres.str.split('|')
    movies['title'] = movies['title'].apply(lambda x: x.lower())
    #ratings.drop('timestamp', axis=1, inplace=True)
    #movies.drop('genres', axis=1, inplace=True)


    # Subset of the data
    movies_subset = movies[:subset_size]
    
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    recommended_movies = []
    #tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    data['keywords'] = data['keywords'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['keywords'])
    consine_sm = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(data.index,index=data['title']).drop_duplicates()
    
    for title in movie_list:
        idx = indices[title]
        sim_scores = list(enumerate(consine_sm[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movies_indices = [i[0] for i in sim_scores]
        recommended_movies.append(data['title'].iloc[movies_indices].values.tolist())
    
    return recommended_movies
