import streamlit as st
import pickle
import pandas as pd

def recommend_movies(movie_title, movies_df, similarity):
    # Find movie index
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in movies_list:
            recommendations.append(movies_df.iloc[i[0]]['title'])
        
        return recommendations
    except:
        return []

# Load data
@st.cache_data
def load_data():
    try:
        with open('new_df.pkl', 'rb') as f:
            movies_df = pickle.load(f)
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        return movies_df, similarity
    except:
        return None, None

st.title("Movie Recommender System")

movies_df, similarity = load_data()

if movies_df is not None:
    selected_movie = st.selectbox("Choose a movie:", movies_df['title'].values)
    
    if st.button("Recommend"):
        recommendations = recommend_movies(selected_movie, movies_df, similarity)
        
        if recommendations:
            st.write("Recommended Movies:")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.error("Could not find recommendations")
else:
    st.error("Please run the notebook first to generate data files")