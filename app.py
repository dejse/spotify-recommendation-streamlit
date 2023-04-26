import streamlit as st
import utils
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import utils

# Setup
data_path = "./data"

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
st.markdown("# Spotify Song Recommender")

st.markdown("## Song Details")
st.markdown("Here are the details of the song you selected:")
st.dataframe(utils.show_song_details(con, id))

st.markdown("## Recommendations")
recommendations = utils.make_recommendations(con,id)
st.dataframe(recommendations)


try:
    iframes = ""
    track_ids = recommendations["id"].to_list()
    for i in track_ids:
        src = f"https://open.spotify.com/embed/track/{i}"
        #iframes += f"""
        #<iframe style="" src="https://open.spotify.com/embed/track/{i}" frameBorder="1" 
        #    allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
        #"""
        st.components.v1.iframe(src, width=180, height=90)
except:
    pass