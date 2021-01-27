# %%
import json

from transat_rl.env.constants import NPI_COLUMNS, OXFORD_CSV_PATH, NPI_BOUNDS_PATH
from transat_rl.env.utils import load_dataset

# %%
df = load_dataset(OXFORD_CSV_PATH)
# %%
df = df[NPI_COLUMNS]
max = df.max()
min = df.min()
# %%
npi_bounds = {npi: (min[npi], max[npi]) for npi in df.columns}
# %%
print(npi_bounds)

# %%
with open(NPI_BOUNDS_PATH, "w") as f:
    json.dump(npi_bounds, f)

# %%
