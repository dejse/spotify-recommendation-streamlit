from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import duckdb
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

data_path = "./data"

def clean_db(con: duckdb.DuckDBPyConnection):
    con.execute("drop table if exists albums")
    con.execute("drop table if exists artists")
    con.execute("drop table if exists tracks")
    con.execute("drop table if exists genres")
    con.execute("drop table if exists features")
    con.execute("drop table if exists features_preprocessed")


def load_data_into_db(con: duckdb.DuckDBPyConnection):
    con.read_csv(f"{data_path}/spotify_albums.csv", all_varchar=False).create("albums")
    con.read_csv(f"{data_path}/spotify_artists.csv", all_varchar=False).create("artists")
    con.read_csv(f"{data_path}/spotify_tracks.csv", all_varchar=False).create("tracks")


def create_genre_table(con: duckdb.DuckDBPyConnection):
    df = con.query("select track_id, genres from artists").pl()
    df = (
        # Explode genre array into rows for each genre
        df.with_columns(
            pl.col("genres").str.replace("^\[\]$", "NoGenre").str.replace_all("hip hop", "hiphop")
        ).with_columns(
            pl.col("genres").str.strip("[]").str.replace_all("'", "").str.split(", ")
        )
        .explode("genres")

        # Explode each genre into words
        .with_columns(
            pl.col("genres").str.split(" ").alias("words")
        )
        .explode("words")
    )

    # Get top X genre words by count (like "pop" or "rock")
    top_genre = (
        df
        .groupby("words")
        .agg([
            # word count is also the ranking
            pl.count("words").alias("count"),
        ])
        .filter(pl.col("words") != "NoGenre")
        .sort("count", descending=True)
        .limit(20)
    )

    # final genre df
    genre_final = (
        df
        .join(top_genre, on="words", how="left")
        .with_columns(
            pl.when(pl.col("count").is_null()).then("Other").otherwise(pl.col("words")).alias("genre_class")
        )
        .sort("count", descending=True)
        .unique(subset="track_id", keep="first")
    )
    
    # Back to DuckDB
    sql = """ 
        drop table if exists genres;
        create table genres as 
        select track_id, genres, genre_class from genre_final
    """
    con.execute(sql)


def create_features_table(con: duckdb.DuckDBPyConnection):
    # Final Table with all features and joins 
    con.execute("""
        drop table if exists final;
        create table features as
        with final as (
            select
                t.id, 
                t.acousticness, t.danceability, t.energy, t.instrumentalness, t.liveness, t.loudness, t.speechiness, t.tempo, t.valence,
                t.name as track_name, ar.name as artist_name, a.name as album_name, concat(ar.name, ' - ', t.name) as song_detail
                coalesce(g.genres, 'Other') as genres, 
                coalesce(g.genre_class, 'Other') as genre_class,
            from tracks t
            join albums a on t.album_id = a.id
            join artists ar on a.artist_id = ar.id
            left join genres g on t.id = g.track_id
            order by t.id
        )
        select *, row_number() over (order by id) -1 as row_number
        from final
        --where genre_class != 'Other'
    """)


def create_features_processed_table(con: duckdb.DuckDBPyConnection):
    # Define X and y
    df = con.query("select * from features order by row_number").df()
    cols = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    y = df["genre_class"]
    X = df[cols]

    # Scale X
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode y
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Concat DataFrames
    df_preprocessed = pd.concat([pd.DataFrame(X_scaled, columns=cols), pd.DataFrame(y_encoded, columns=["genre_class"])], axis=1)
    df_preprocessed = pd.concat([df[["id"]], df_preprocessed], axis=1)
  
    # Write to DuckDB
    con.execute("""
        drop table if exists features_preprocessed;
        create table features_preprocessed as
            select *, row_number() over (order by id) - 1 as row_number
            from df_preprocessed
            order by id
    """)


def get_model_knn() -> KNeighborsClassifier:
    # Check if model exists
    if Path(f"{data_path}/knn.pkl").exists():
        with open(Path(f"{data_path}/knn.pkl"), "rb") as f:
            return pickle.load(f)

    # Otherwise, create model
    df = con.query("select * from features_preprocessed order by id").df()
    X = df.drop(["id", "genre_class", "row_number"], axis=1)
    y = df["genre_class"]
    
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree")
    knn.fit(X,y)
    
    with open(Path(f"{data_path}/knn.pkl"), "wb") as f:
        pickle.dump(knn, f)

    return knn


def lookup_song(con: duckdb.DuckDBPyConnection, lookup_query: str = "", genre_class: str = "") -> pd.DataFrame:
    sql = """
      select *
      from features
      where 
        regexp_matches(lower(song_detail), left($query, 50))
        AND 
        regexp_matches(lower(genre_class), $genre)
      order by id
      limit 30
    """
    return con.execute(sql, { "query": lookup_query.lower(), "genre": genre_class.lower() }).fetch_df()


def show_song_details(con: duckdb.DuckDBPyConnection, track_id: str) -> str:
    sql = """select * from features where id = $track_id"""
    return con.execute(sql, { "track_id": track_id}).fetch_df()


def make_recommendations(con: duckdb.DuckDBPyConnection, track_id: str, n_neighbors: int = 2000) -> pd.DataFrame:
    # Get test data
    sql = """select * from features_preprocessed where id = $track_id order by id"""
    df_test = con.execute(sql, { "track_id": track_id}).fetch_df()
    X_test = df_test.drop(["id", "genre_class", "row_number"], axis=1)
    y_test = df_test["genre_class"]

    # Predict
    knn = get_model_knn()
    distances, indices = knn.kneighbors(X_test, n_neighbors=n_neighbors)

    # Lookup song details
    recommended = pd.DataFrame({"row_number": indices[0], "distance": distances[0]})
    sql = """
        select * from features
        where row_number in (select row_number from recommended)
    """
    details = con.query(sql).df()
    merged = pd.merge(recommended, details, left_on="row_number", right_on="row_number")
    
    merged = merged.filter(
        ["id", "song_detail", "genres", "genre_class",
          "row_number", "distance",
         "acousticness", "danceability", "energy", "instrumentalness",
         "liveness", "loudness", "speechiness", "tempo", "valence"]
    )

    genre = merged.iloc[0]["genre_class"]
    merged = merged.query("genre_class == @genre").head(10)
    return merged



if __name__ == "__main__":
    con = duckdb.connect(database=f"{data_path}/spotify.db")
    con.close()