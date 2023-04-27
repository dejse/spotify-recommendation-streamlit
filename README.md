# spotify-recommendation-streamlit

Project for MS in DS

## Installation

```sh
conda create -n spotify_env python=3.11
conda activate spotify_env
pip install -r streamlit_req

conda activate spotify_dagster_env python=3.11
pip install -r dagster_req
```

## Run

```sh
conda activate spotify_env
python -m streamlit run app.py


conda activate spotify_dagster_env
dagster dev
```