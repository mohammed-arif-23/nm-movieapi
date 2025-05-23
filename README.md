# AI Movie Recommender

A Streamlit app that recommends movies based on genre similarity and release year.

## Features

- Search for a movie by title
- Get recommendations based on genre similarity and release year
- Display movies in a grid with information about each movie

## How it works

1. Load movie data from a CSV file
2. Preprocess the data by extracting the year from the title and converting genres to a list
3. Use cosine similarity to find movies with similar genres
4. Filter results to movies released within 5 years of the selected movie
5. Display the results in a grid with information about each movie

## Requirements

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Requests
- Pillow

## Usage

1. Install the required packages with `pip install -r requirements.txt`
2. Run the app with `streamlit run movie_recommender_app.py`
3. Open a web browser and navigate to `http://localhost:8501`

## Data

The app uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, which is a collection of movie ratings and metadata. The dataset is included in the repository as a CSV file named `movies.csv`.

## License

This app is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
