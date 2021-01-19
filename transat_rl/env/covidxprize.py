import os
import sys

import gym
import numpy as np
import pandas as pd
from gym import spaces
from transat_rl.env.constants import NPI_COLUMNS
from transat_rl.env.utils import load_dataset, preprocess_historical_basic

here = os.path.dirname(os.path.abspath(__file__))
python_exe = sys.executable

# Prescriptor per country
# Stadard predictor available at https://github.com/leaf-ai/covid-xprize/tree/master/covid_xprize/standard_predictor
# predictor API: python predict.py -s start_date -e end_date -ip path_to_ip_file -o path_to_output_file

DAY = pd.Timedelta(1, unit="day")
COLS = ["country_name", "region_name", "Date"] + NPI_COLUMNS


class CovidBaselineEnv(gym.Env):
    """Custom COVID scenario for NPI prescription"""

    metadata = {"render.modes": ["human"]}
    N_npis: int = 12
    base_columns = ["CountryName", "RegionName", "Date"]

    def __init__(
        self,
        predictor_script_path: str,
        oxford_csv_path: str,
        future_days: int = 180,
        lookback_days: int = 21,
        geoids: list = ["France__"],
    ):
        super().__init__()

        # Every geoid condidered in the Challenge
        self.geoids = geoids
        # Load all data necessary for the environment: cumulative cases, new cases, new cases 7-smooth, NPIs
        self.data = preprocess_historical_basic(oxford_csv_path, geoids)

        # self.weights_data = pd.read_csv(
        #     weights_file, sep=",", dtype={"country_name": str, "region_name": str}
        # ).fillna("")

        # Add unique GeoID
        # self.weights_data["GeoID"] = (
        #     self.weights["country_name"] + "__" + self.weights["region_name"].astype(str)
        # )

        # Episode attributes
        # Dynamic attributes
        self.geoid_episode = None
        self.country_name, self.region_name = None, None
        self.data_episode = None
        self.df_npis_episode = None
        self.start_index = None
        self.t = None  # current step
        self.weights = np.zeros((self.N_npis,))

        # Static attributes
        self.lookback_days = lookback_days
        self.future_days = future_days
        self.T = 1  # episode length for simulation

        # Output management
        self.predictor_inp_path = os.path.join(here, "predictor_input.csv")
        self.predictor_out_path = os.path.join(here, "predictor_output.csv")
        self.predictor_exe_template = f"{predictor_script_path} -s {'{}'} -e {'{}'} -ip {self.predictor_inp_path} -o {self.predictor_out_path}"

        # Trace of predictor-prescriptor
        self.history = None

        # ACTION SPACE
        # Each NPI has a value between 0 anf 4 per timestep --> 5 discrete actions per country
        self.action_space = spaces.Box(low=0, high=4, dtype=int, shape=(self.N_npis,))

        # OBSERVATION SPACE: a window of past new cases
        self.observation_space = spaces.Box(
            low=0, high=100000000, dtype=int, shape=(self.lookback_days,)
        )

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = True  # only one step per episode
        start_index = self.start_index
        end_index = self.start_index + self.future_days - 1

        # Overwrite ip file with action, save csv to call predict.py
        i = len(self.df_npis_episode)
        sample_row = self.df_npis_episode.iloc[i-1].tolist()
        while end_index >= len(self.df_npis_episode):
            sample_row = sample_row[:]
            sample_row[2] +=  DAY
            self.df_npis_episode.loc[i] = sample_row
            i += 1

        self.df_npis_episode.loc[start_index:end_index, NPI_COLUMNS] = action

        self.df_npis_episode.to_csv(self.predictor_inp_path, index=False)

        # Run predictor
        pred_start_date = self.df_npis_episode.Date[start_index]
        pred_end_date = self.df_npis_episode.Date[end_index]

        start_date_str = pred_start_date.strftime("%Y-%m-%d")
        end_date_str = pred_end_date.strftime("%Y-%m-%d")

        predictor_exe = self.predictor_exe_template.format(start_date_str, end_date_str)
        # print("Calling: ", predictor_exe)
        os.system(f"{python_exe} {predictor_exe}")

        # Read predictions
        df_predictions = load_dataset(self.predictor_out_path)
        future_new_cases = df_predictions.PredictedDailyNewCases.to_numpy()
        # print("len future new cases: ", len(future_new_cases))
        # new observation
        obs = future_new_cases[: self.lookback_days]

        # compute reward
        reward = self.compute_reward(
            np.array(action),
            self.weights,
            np.array(future_new_cases),
        )

        # update history
        if self.t + 1 < self.T:
            self.history["observations"][self.t + 1] = obs
        self.history["new_cases"][self.t, :] = future_new_cases
        self.history["reward"][self.t] = reward
        self.history["prescriptions"][self.t] = action

        # update step in episode
        self.t += 1

        return obs, reward, done, self.history

    def compute_reward(
        self, npis: np.ndarray, weights: np.ndarray, new_cases: np.ndarray
    ):
        economic_proxy = npis.T @ weights
        gamma = 1
        cum_cases = new_cases.sum()
        return -(economic_proxy + gamma * cum_cases)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.geoid_episode = np.random.choice(self.geoids)
        self.country_name, self.region_name = self.geoid_episode.split("__")

        self.data_episode = self.data[
            (self.data.CountryName == self.country_name)
            & (self.data.RegionName == self.region_name)
        ]
        self.data_episode.reset_index(level=0, drop=True, inplace=True)

        self.df_npis_episode = self.data_episode[self.base_columns + NPI_COLUMNS]

        d = self.lookback_days
        self.start_index = np.random.randint(len(self.data_episode) - self.T - d) + d
        self.t = 0  # current step

        # retrieve "self.weights" corresponding to "self.geoid_episode"

        data_previous_days = self.data_episode.iloc[
            self.start_index - self.lookback_days : self.start_index
        ]
        obs = data_previous_days["NewCasesSmoothed7Days"].to_numpy()

        self.history = {
            "geoid": self.geoid_episode,
            "observations": np.zeros((self.T, self.lookback_days)),
            "new_cases": np.zeros((self.T, self.future_days)),
            "reward": np.zeros(self.T),
            "prescriptions": np.zeros((self.T, self.N_npis)),
        }

        self.history["observations"][0, :] = obs

        return obs

    def render(self, mode="human", close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
            return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
                the list of supported modes. It's recommended to call super()
                in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        ...