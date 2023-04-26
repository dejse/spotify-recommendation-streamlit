import streamlit as st
import utils
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import utils

# Setup
data_path = Path("data")
data_path_string = data_path.resolve().as_posix()

@st.cache_resource
def get_db():
    return duckdb.connect(database=f"{data_path}/spotify.db")

con = get_db()


# Sidebar
genres = con.query("""
    select genre_class, count(genre_class) 
    from features 
    group by genre_class 
    order by count(genre_class) desc
    """).df()["genre_class"].to_list()
genres = [""] + genres

st.sidebar.title("Choose your Song")

# Find Song and ID
try:
    selected_genre_class = st.sidebar.selectbox("Which genre?", genres)
    query = st.sidebar.text_input("Search for a song or artist", "")
    songs = utils.lookup_song(con, query, selected_genre_class)
    selected_song = st.sidebar.selectbox("Select Song", songs["song_detail"])[:20]
    id = songs.query("`song_detail`.str.contains(@selected_song)", engine="python").iloc[0]["id"]
except:
    id = ""


# Main
st.title("Spotify Song Recommender")
recommendations = utils.make_recommendations(con,id)
st.dataframe(recommendations)

