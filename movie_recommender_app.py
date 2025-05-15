import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('movies.csv')
        df['genres'] = df['genres'].fillna('')
        df['processed_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True).str.strip().str.lower()
        df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
        df['genres_list'] = df['genres'].str.split('|')
        df['color'] = [f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}" for _ in range(len(df))]
        return df
    except FileNotFoundError:
        st.error("Make sure 'movies.csv' is in the same directory as the script.")
        st.stop()

movies_df = load_data()

def get_recommendations_cosine(title, movies_df=movies_df):
    try:
        title = title.strip().lower()
        movie_row = movies_df[movies_df['processed_title'] == title].iloc[0]
        index = movie_row.name
        selected_movie_year = movie_row['year']
        selected_movie_genres = movie_row['genres_list']

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        valid_recommendations = []
        for i, similarity in sim_scores[1:]:
            recommended_movie_year = movies_df.iloc[i]['year']
            recommended_movie_genres = movies_df.iloc[i]['genres_list']
            if (selected_movie_year - 5 <= recommended_movie_year <= selected_movie_year + 5):
                if any(genre in selected_movie_genres for genre in recommended_movie_genres):
                    valid_recommendations.append((i, similarity))
                    
        if not valid_recommendations:
            return pd.DataFrame()
        
        top_similar_indices = [i for i, _ in valid_recommendations[:10]]
        return movies_df.iloc[top_similar_indices]
    except IndexError:
        return pd.DataFrame()
    except KeyError:
        return pd.DataFrame()

def display_movies_grid(movie_data, header=""):
    if movie_data.empty:
        st.warning("No movies to display.")
        return

    st.subheader(header)
    num_cols = 5
    movies_to_show = movie_data.head(10)
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

