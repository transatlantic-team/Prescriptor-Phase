import os
from pprint import pprint

from tensorforce import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from transat_rl.env.utils import download_csv
from transat_rl.env.constants import LATEST_DATA_URL, OXFORD_CSV_PATH
from transat_rl.env.covid import CovidEnv

data_folder = os.path.dirname(OXFORD_CSV_PATH)

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

# number of prescriptions (1 prescription per day)
future_days = 3

# Path to the "standard_predictor/predict.py" from the covid-xprize repo
# the covid-xprize package needs to be installed "pip install -e."
predictor_script_path = "/Users/romainegele/Documents/xPrize/covid-xprize/covid_xprize/standard_predictor/predict.py"

# Instanciate environment and wrap it up in Tensorforce.Environment class
env = Environment.create(
    CovidEnv(future_days, predictor_script_path, OXFORD_CSV_PATH),
    max_episode_timesteps=future_days
)

print("ACTION SPACE")
pprint(env.actions())

print("STATE SPACE")
pprint(env.states())

# Create Agent
agent = Agent.create(
    agent='ppo', environment=env, batch_size=10, learning_rate=1e-3
)

# Create a runner
runner = Runner(
    agent=agent,
    environment=env,
)

runner.run(num_episodes=2)

