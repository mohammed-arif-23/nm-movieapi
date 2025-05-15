import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the MovieLens dataset (movies.csv)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('movies.csv')
        # Preprocess the data
        df['genres'] = df['genres'].fillna('')
        df['processed_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True).str.strip().str.lower()
        df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
        df['genres_list'] = df['genres'].str.split('|')
        # Create a poster placeholder color
        df['color'] = [f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}" for _ in range(len(df))]
        return df
    except FileNotFoundError:
        st.error("Make sure 'movies.csv' is in the same directory as the script.")
        st.stop()

# Load data
try:
    movies_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Function to get movie recommendations based on cosine similarity
def get_recommendations_cosine(title, movies_df=movies_df):
    try:
        title = title.strip().lower()
        # Find the movie.
        movie_row = movies_df[movies_df['processed_title'] == title].iloc[0]
        index = movie_row.name
        selected_movie_year = movie_row['year']
        selected_movie_genres = movie_row['genres_list']

        # Create a TF-IDF Vectorizer for the 'genres' column
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

        # Calculate cosine similarity between movies
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        sim_scores = list(enumerate(cosine_sim[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get movies within the year range and same genres
        valid_recommendations = []
        for i, similarity in sim_scores[1:]:  # Exclude the movie itself
            recommended_movie_year = movies_df.iloc[i]['year']
            recommended_movie_genres = movies_df.iloc[i]['genres_list']
            
            # Check if the movie is within the year range (past 5 or future 5 years)
            if (selected_movie_year - 5 <= recommended_movie_year <= selected_movie_year + 5):
                # Check if the movie shares at least one genre
                if any(genre in selected_movie_genres for genre in recommended_movie_genres):
                    valid_recommendations.append((i, similarity))
                    
        if not valid_recommendations: # Check if valid_recommendations is empty
            return pd.DataFrame()
        
        top_similar_indices = [i for i, _ in valid_recommendations[:10]]
        return movies_df.iloc[top_similar_indices]

    except IndexError:
        return pd.DataFrame()  # Return an empty DataFrame if the movie is not found
    except KeyError:
        return pd.DataFrame()

# Display movies in a grid (2 rows x 5 columns)
def display_movies_grid(movie_data, header=""):
    if movie_data.empty:
        st.warning("No movies to display.")
        return

    st.subheader(header)
    num_cols = 5
    movies_to_show = movie_data.head(10)  # Show up to 10 movies (2 rows)
    rows = [movies_to_show.iloc[i:i+num_cols] for i in range(0, len(movies_to_show), num_cols)]
    for row in rows:
        cols = st.columns(num_cols)
        for idx, (index, movie) in enumerate(row.iterrows()):
            with cols[idx]:
                st.markdown(
                    f"<div style='width:auto;height:120px;background-color:{movie['color']};border-radius:10px;margin-bottom:8px;display:flex;align-items:center;justify-content:center;'>{movie['title']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**{movie['title']}**")
                st.caption(f"Year: {int(movie['year']) if not pd.isna(movie['year']) else 'N/A'}")
                st.caption(f"Genres: {movie['genres']}")
    st.markdown("---")

def main():
    st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 AI-Powered Movie Recommendation System")
    st.write("Welcome! Enter a movie title to get recommendations based on genre similarity and release year.")

    search_term = st.text_input("🔎 Search for a movie:", "")

    if search_term:
        with st.spinner("Searching for movies and generating recommendations..."):
            filtered_movies = movies_df[movies_df['processed_title'].str.contains(search_term.lower(), case=False)]
            if not filtered_movies.empty:
                display_movies_grid(filtered_movies, header=f"Movies matching '{search_term}':")
                # Get the first movie from the filtered list to use as a base for recommendations
                selected_movie = filtered_movies.iloc[0]['processed_title']
                recommendations = get_recommendations_cosine(selected_movie)
                if not recommendations.empty:
                    display_movies_grid(recommendations, header=f"Recommendations for '{filtered_movies.iloc[0]['title']}':")
                else:
                    st.info("No recommendations found for this movie based on the criteria.")
            else:
                st.warning(f"No movies found matching '{search_term}'. Please try again.")
    else:
        st.write("Please enter a movie title to see recommendations.")

if __name__ == "__main__":
    main()
