import streamlit as st
import utils
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pickle

# Setup
data_path = Path("./data")
data_path_string = data_path.resolve().as_posix()

@st.experimental_singleton
def get_db():
    return duckdb.connect(database=f"{data_path}/spotify.db")

@st.experimental_memo
def load_data(con):
    return utils.load_data_into_pd(con)

con = get_db()
df = load_data(con)


# Sidebar
st.sidebar.title("Choose your Song")


# Main
st.title("Spotify Song Recommender")