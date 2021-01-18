import os
import numpy as np
import pandas as pd

from transat_rl.env.utils import download_csv, load_NPIs_filtered
from transat_rl.env.constants import LATEST_DATA_URL

from transat_rl.env.covid import CovidEnv

here = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(here, "data")
print(here)

download = False
if download:
    # download list of regions where we are evaluated
    download_csv(
        "https://raw.githubusercontent.com/leaf-ai/covid-xprize/master/countries_regions.csv",
        "kept_regions",
        dest_folder=data_folder,
    )

    # download latest version of Oxford dataset
    download_csv(LATEST_DATA_URL, "OxCGRT_latest", dest_folder=data_folder)

oxford_csv_path = os.path.join(data_folder, "OxCGRT_latest.csv")
input_npis_df = load_NPIs_filtered(
    oxford_csv_path,
    os.path.join(data_folder, "kept_regions.csv"),
)
input_npis_path = os.path.join(here, "data", "historic_NPIs_lastest.csv")
input_npis_df.to_csv(input_npis_path, index=False)

lookback_days = 10
future_days = 10
predictor_script_path = "/Users/romainegele/Documents/xPrize/covid-xprize/covid_xprize/standard_predictor/predict.py"

env = CovidEnv(
    lookback_days, future_days, predictor_script_path, oxford_csv_path
)

EPISODES = 1

for episode in range(EPISODES):

    obs = env.reset()
    done = False
    t = 0
    while not(done):
        print(f"t={t}")
        action = [[0 for _ in range(12)]]
        obs, rew, done, info = env.step(action)
        print(f"obs: {obs}")
        print(f"rew: {rew}")
        print(f"done: {done}")
        t += 1

    break