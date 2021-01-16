import gym
from gym import spaces
import pandas as pd
import numpy as np

class COVID(gym.Env):
  """Custom COVID scenario for NPI prescription"""
  metadata = {'render.modes': ['human']}

  def __init__(self, weights_file, episode_length, predictor, lookback_days):
    super(COVID, self).__init__()

    # Episode length for simulation
    self.T = episode_length

    # Lookback window size
    self.T_context = lookback_days

    # Load costs per region/NPI
    self.weights = pd.read_csv(weights_file, sep=',',
                                dtype={"CountryName": str,
                                    "RegionName": str}).fillna('')

    # Add unique GeoID
    self.weights['GeoID'] = self.weights['CountryName'] + '__' + self.weights['RegionName'].astype(str)

    # Save predictor
    self.predictor = predictor

    ###### ACTION SPACE ######
    # Each NPI has a value between 0 anf 4 per timestep --> 5 discrete actions per country

    self.npi_list = [c for c in self.weights.columns if c not in ['CountryName', 'RegionName', 'GeoID']]
    self.action_space = spaces.Box(low=0, high=4, dtype=np.uint8, shape=(len(self.npi_list),))

    ###### OBSERVATION SPACE ######
    # State of the system is past (NPIs, cases)

    spaces = {
        'previous_NPIs': spaces.Box(low=0, high=4, dtype=np.uint8, shape=(len(self.npi_list), self.T_context)),
        'previous_R': spaces.Box(low=0.0, high=2.0, dtype=np.float32, shape=(self.T_context, ))

    }

    self.observation_space = spaces.dict.Dict(spaces)


    

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...