import os
import sys

import gym
import numpy as np
import pandas as pd
from gym import spaces
from transat_rl.env.constants import NPI_COLUMNS

here = os.path.dirname(os.path.abspath(__file__))
python_exe = sys.executable

# Prescriptor per country
# Stadard predictor available at https://github.com/leaf-ai/covid-xprize/tree/master/covid_xprize/standard_predictor
# predictor API: python predict.py -s start_date -e end_date -ip path_to_ip_file -o path_to_output_file

DAY = pd.Timedelta(1, unit="day")
COLS = ["country_name", "region_name", "Date"] + NPI_COLUMNS


class CovidEnv(gym.Env):
    """Custom COVID scenario for NPI prescription"""

    metadata = {"render.modes": ["human"]}
    N_npis: int = 12

    def __init__(
        self,
        lookback_days: int,
        future_days: int,
        predictor_script_path: str,
        start_date: str,
        geoid: str,
        input_npis_path: str,
        oxford_data_path: str,
    ):
        super().__init__()

        self.start_date = pd.to_datetime(start_date, format="%Y-%m-%d")

        self.country_name, self.region_name = geoid.split("__")

        self.T = future_days  # episode length for simulation
        self.t = 0  # current step

        self.lookback_days = lookback_days

        # Load costs per region/NPI

        self.weights = np.zeros((self.N_npis,))
        # self.weights = pd.read_csv(
        #     weights_file, sep=",", dtype={"country_name": str, "region_name": str}
        # ).fillna("")

        # Add unique GeoID
        # self.weights["GeoID"] = (
        #     self.weights["country_name"] + "__" + self.weights["region_name"].astype(str)
        # )

        # Save predictor
        self.end_date = None
        self.path_to_ip_file = os.path.join(here, "env_input_ips.csv")
        self.df_npis = pd.read_csv(
            input_npis_path,
            parse_dates=["Date"],
            encoding="ISO-8859-1",
            dtype={"CountryName": str, "RegionName": str},
            error_bad_lines=False,
        )
        self.df_npis.RegionName.fillna("", inplace=True)
        self.df_npis = self.df_npis[(self.df_npis.CountryName == self.country_name)]
        self.df_npis = self.df_npis[(self.df_npis.RegionName == self.region_name)]
        self.path_to_output_file = os.path.join(here, "env_output_cases.csv")
        self.predictor_exe_template = f"{predictor_script_path} -s {'{}'} -e {'{}'} -ip {self.path_to_ip_file} -o {self.path_to_output_file}"

        self.df_oxford = pd.read_csv(oxford_data_path, )
        self.df_oxford = self.df_oxford[(self.df_npis.CountryName == self.country_name)]
        self.df_oxford = self.df_oxford[(self.df_npis.RegionName == self.region_name)]
        self.history = {
            "new_cases": []
        }

        ###### ACTION SPACE ######
        # Each NPI has a value between 0 anf 4 per timestep --> 5 discrete actions per country
        # self.npi_list = [
        #     c
        #     for c in self.weights.columns
        #     if c not in ["country_name", "region_name", "GeoID"]
        # ]

        self.action_space = spaces.Box(low=0, high=4, dtype=int, shape=(1, self.N_npis))

        ###### OBSERVATION SPACE ######
        # State of the system is past (NPIs, cases)

        space_dict = {
            "previous_NPIs": spaces.Box(
                low=0, high=4, dtype=int, shape=(self.N_npis, self.lookback_days)
            ),
            "previous_new_cases": spaces.Box(
                low=0, high=np.inf, dtype=int, shape=(self.lookback_days,)
            ),
        }

        self.observation_space = spaces.dict.Dict(space_dict)

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
        # action: [0..4]^12
        # python predict.py -s start_date -e end_date -ip path_to_ip_file -o path_to_output_file
        # self.predictor_script = "predict.py"

        # self.path_to_ip_file = os.path.join(here, "env_input_ips.csv")
        # self.path_to_output_file = os.path.join(here, "env_output_cases.csv")

        # Start date and end date of NPIs (and predictions)
        pred_start_date = self.start_date + self.t * DAY
        pred_end_date = pred_start_date

        # Row format for ip file
        action_date = pred_start_date
        action_data = [self.country_name, self.region_name, action_date, *action[0]]

        # Overwrite ip file with action, save csv to call predict.py

        print(self.df_npis.columns)
        print("action date: ", action_date)
        print("len action data: ", len(action_data))
        print(action)
        print(self.df_npis[self.df_npis.Date == action_date])

        self.df_npis[self.df_npis.Date == action_date] = action_data
        print("AFTER")
        print(self.df_npis[self.df_npis.Date == action_date])

        self.df_npis.to_csv(self.path_to_ip_file, index=False)

        # Run predictor
        start_date_str = pred_start_date.strftime("%Y-%m-%d")
        end_date_str = pred_end_date.strftime("%Y-%m-%d")

        print(start_date_str, end_date_str)

        predictor_exe = self.predictor_exe_template.format(start_date_str, end_date_str)
        print("Calling: ", predictor_exe)
        os.system(f"{python_exe} {predictor_exe}")

        # self.path_to_output_file

        # Read predictions

        df_predictions = pd.read_csv(
            self.path_to_output_file,
            parse_dates=["Date"],
            encoding="ISO-8859-1",
            dtype={"CountryName": str, "RegionName": str},
            error_bad_lines=False,
        )
        new_cases = df_predictions.PredictedDailyNewCases.to_numpy()

        print("new_cases")


        info = {}
        observation = None
        done = None

        npis = None
        new_cases = None
        reward = None  # self.compute_reward(npis, self.weights, new_cases)

        return observation, reward, done, info

    def compute_reward(
        self, npis: np.ndarray, weights: np.ndarray, new_cases: np.ndarray
    ):
        economic_proxy = npis.T @ weights
        gamma = 1
        r0 = (new_cases[-1] - new_cases[-2]) / new_cases[-1] + 1
        return -(economic_proxy + gamma * r0)

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
        ...

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
