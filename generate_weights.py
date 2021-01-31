import os
import pandas as pd
import numpy as np
from transat_rl.env.constants import NPI_COLUMNS

here = os.path.dirname(os.path.abspath(__file__))

def generate_weights(zeros=False, ones=False):
    
    df = pd.read_csv(os.path.join(here, "data", "kept_regions.csv"))
    n_regions = len(df)
    for npi in NPI_COLUMNS:
        if zeros:
            df[npi] = np.zeros(n_regions)
        elif ones:
            df[npi] = np.ones(n_regions)
        else:
            df[npi] = np.random.random_sample(n_regions)

    # Normalize s.t each regions sums to number of NPIs
    if(not zeros and not ones):
        df[NPI_COLUMNS] = df[NPI_COLUMNS].div(df[NPI_COLUMNS].sum(axis=1), axis=0)
        df[NPI_COLUMNS] = df[NPI_COLUMNS].mul(len(NPI_COLUMNS))
    
    return df


if __name__ == "__main__":
    df = generate_weights()
    print(df.head())
