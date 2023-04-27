import streamlit as st
import utils
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import utils


### Setup
@st.cache_resource
def get_db(data_path: str = "./data") -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=f"{data_path}/spotify.db")

con = get_db()



### Sidebar
st.sidebar.title("Choose your Song")
genres = utils.get_genre_list(con)


try:
    selected_genre = st.sidebar.selectbox("Which genre?", genres)
    query = st.sidebar.text_input("Search for a song or artist", "")
    songs = utils.lookup_songs(con, query, selected_genre)
    selected_song = st.sidebar.selectbox("Select Song", songs["song_detail"])[:20]
    track_id = songs.query("`song_detail`.str.contains(@selected_song)", engine="python").iloc[0]["id"]
except:
    track_id = ""



### Main
st.markdown("# Spotify Song Recommender")
st.markdown("## Song Details")
st.markdown("Here are the details of the song you selected:")
st.dataframe(utils.show_song_details(con, track_id))

st.markdown("## Recommendations")
recommendations = utils.make_recommendations(con, track_id)
st.dataframe(recommendations)


try:
    iframes = ""
    track_ids = recommendations["id"].to_list()
    for index, id in enumerate(track_ids):
        src = f"https://open.spotify.com/embed/track/{id}"
        st.components.v1.iframe(src, width=180, height=90)
except:
    pass