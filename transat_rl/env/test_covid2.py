import os
from pprint import pprint

import numpy as np
import pandas as pd

from transat_rl.env.utils import download_csv, load_NPIs_filtered
from transat_rl.env.constants import LATEST_DATA_URL, OXFORD_CSV_PATH
from transat_rl.env.covid2 import CovidEnv

here = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(here, "data")

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


predictor_script_path = "/Users/romainegele/Documents/xPrize/covid-xprize/covid_xprize/standard_predictor/predict.py"

env = CovidEnv(predictor_script_path, OXFORD_CSV_PATH, future_days=2, lookback_days=4, episode_length=4)

EPISODES = 1

for episode in range(EPISODES):

    obs = env.reset()
    print("first obs: ", obs)
    done = False
    t = 0
    while not (done):
        print(f"t={t}")
        action = [1 for _ in range(12)]
        obs, rew, done, info = env.step(action)
        # print(f"obs: {obs}")
        print(f"rew: {rew}")
        # print(f"done: {done}")
        t += 1

    # break

# print("<- HISTORY ->")
# pprint(env.history)
print(env.history["reward"])