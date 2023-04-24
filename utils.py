"""
    All extracted from E2E.ipynb
"""

from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

data_path = Path("../data")
data_path_string = data_path.resolve().as_posix()
con = duckdb.connect(database=f"{data_path}/spotify.db")

def clean_db(con: duckdb.DuckDBPyConnection):
    con.execute("drop table if exists lyrics_features")
    con.execute("drop table if exists low_level_audio_features")
    con.execute("drop table if exists albums")
    con.execute("drop table if exists artists")
    con.execute("drop table if exists tracks")
    con.execute("drop table if exists features")
    con.execute("drop view if exists lookup")


def load_data_into_db(con: duckdb.DuckDBPyConnection, data_path_string: str):
    con.read_csv(f"{data_path_string}/lyrics_features.csv", all_varchar=False).create("lyric_features")
    con.read_csv(f"{data_path_string}/low_level_audio_features.csv", all_varchar=False).create("low_level_audio_features")
    con.read_csv(f"{data_path_string}/spotify_albums.csv", all_varchar=False).create("albums")
    con.read_csv(f"{data_path_string}/spotify_artists.csv", all_varchar=False).create("artists")
    con.read_csv(f"{data_path_string}/spotify_tracks.csv", all_varchar=False).create("tracks")


def create_lookup_table(con: duckdb.DuckDBPyConnection):
    sql = """
        drop view if exists lookup;
        create view lookup as
        select 
            t.id, t.name as track_name, ar.name as artist_name, a.name as album_name, 
            t.preview_url, t.track_href, t.analysis_url
        from tracks t
        left join albums a on t.album_id = a.id
        left join artists ar on a.artist_id = ar.id
        """
    con.execute(sql)


def create_features_table(con: duckdb.DuckDBPyConnection):
    sql = """
        drop table if exists features;
        create table features as
        select
            t.id, t.acousticness, t.danceability, t.energy, t.instrumentalness, 
            t.liveness, t.loudness, t.speechiness, t.tempo, t.valence
        from tracks t
        """
    con.execute(sql)


def load_data_into_pd(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.query("select * from features").df()


def get_pipeline(df: pd.DataFrame) -> Pipeline:
    # Check if pipeline exists
    if Path(data_path / "pipeline.pkl").exists():
        with open(data_path / "pipeline.pkl", "rb") as f:
            return pickle.load(f)
    
    # Otherwise, create pipeline
    pipeline = Pipeline([ ("scaler", MinMaxScaler()) ])
    pipeline.fit(df.loc[:, df.columns != "id"])
    with open(data_path / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
        
    return pipeline


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    pipe = get_pipeline(df)
    X = pipe.transform(df.loc[:, df.columns != "id"])
    return X


def get_model_knn(X: np.ndarray or None = None) -> NearestNeighbors:
    # Check if model exists
    if Path(data_path / "knn.pkl").exists():
        with open(data_path / "knn.pkl", "rb") as f:
            return pickle.load(f)

    # Otherwise, create model
    knn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
    knn.fit(X)
    with open(data_path / "knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    return knn


def lookup_ids(con: duckdb.DuckDBPyConnection, lookup_query: str) -> str:
    sql = """
      select *
      from lookup
      where 
        regexp_matches(lower(concat(track_name, ' ', artist_name)), $param)
      limit 10
    """
    return con.execute(sql, { "param": lookup_query.lower()}).fetch_df()


def make_recommendations(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, id: str, n_neighbors: int = 10) -> pd.DataFrame:
    # Predict
    df_test = df.query(f"id == '{id}'")
    X_test = preprocess_data(df_test)
    knn = get_model_knn()
    distances, indices = knn.kneighbors(X_test, n_neighbors=n_neighbors)
    ids = df.loc[indices[0].tolist(), :]

    # Lookup song details
    sql = """
        select * 
        from lookup
        where id in (select id from ids)
    """
    details = con.query(sql).df()

    # Merge
    merge = ids.merge(details, how="left", on="id")
    return merge