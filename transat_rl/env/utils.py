import os
import urllib.request
import shutil
import pandas as pd
import platform

from transat_rl.env.constants import NPI_COLUMNS


def download_csv(CSV_URL, fname, dest_folder="./data"):
    # Local file: base dir, data
    os.makedirs(dest_folder, exist_ok=True)
    DATA_FILE = f"{dest_folder}/{fname}.csv"
    urllib.request.urlretrieve(CSV_URL, DATA_FILE)
    print(f"{fname} updated to {dest_folder}")


def load_dataset(url):
    latest_df = pd.read_csv(
        url,
        parse_dates=["Date"],
        encoding="ISO-8859-1",
        dtype={"RegionName": str, "RegionCode": str},
        error_bad_lines=False,
    )
    latest_df["RegionName"] = latest_df["RegionName"].fillna("")
    return latest_df

def preprocess_historical_basic(oxford_csv_path: str, geoids: list):
    """Create a copy of preprocessed data from OxCGRT dataframe"""

    df = load_dataset(oxford_csv_path)

    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    df["GeoID"] = df["CountryName"] + "__" + df["RegionName"].astype(str)

    # Remove GeoID not considered in the competition
    df = df[df["GeoID"].isin(geoids)]

    # Add new cases column
    df["NewCases"] = df.groupby("GeoID").ConfirmedCases.diff().fillna(0)
    df.loc[df.NewCases < 0, "NewCases"] = 0

    df["NewCasesSmoothed7Days"] = df.groupby("GeoID").NewCases.rolling(7, win_type=None).mean().fillna(0).reset_index(level=0, drop=True) #.reset_index()

    # Keep only columns of interest
    id_cols = ["CountryName", "RegionName", "GeoID", "Date", "ConfirmedCases", "NewCases", "NewCasesSmoothed7Days"]

    df = df[id_cols + NPI_COLUMNS]

    # Fill any missing case values by interpolation and setting NaNs to 0
    df.update(
        df.groupby("GeoID").NewCases.apply(lambda group: group.interpolate()).fillna(0)
    )

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLUMNS:
        df.update(df.groupby("GeoID")[npi_col].ffill().fillna(0))

    return df


def load_NPIs_filtered(
    data_file="./data/OxCGRT_latest.csv", countries_regions_file="./data/kept_regions.csv"
):

    df = load_dataset(data_file)

    df_countries = pd.read_csv(
        countries_regions_file, sep=",", dtype={"CountryName": str, "RegionName": str}
    )

    # Keep only NPI columns
    npi_cols = ["CountryName", "RegionName", "Date", *NPI_COLUMNS]
    df = df[npi_cols]

    df_countries["RegionName"] = df_countries["RegionName"].fillna("")

    # Kept rows from countries_regions_file
    kept_countries = df_countries["CountryName"].unique()

    # Countries with regions have not empty RegionName
    countries_with_regions = df_countries[df_countries["RegionName"] != ""][
        "CountryName"
    ].unique()

    # Countries without regions are the difference between two previous sets
    countries_without_regions = list(set(kept_countries) - set(countries_with_regions))

    # Kept regions include '' for whole country in countries_with_regions
    kept_regions = df_countries["RegionName"].unique()

    # Filter: either dont'have regions --> RegionName is Nan
    #                have regions --> RegionName in kept_regions
    df = df[
        (df["CountryName"].isin(countries_without_regions) & (df["RegionName"] == ""))
        ^ (
            (df["CountryName"].isin(countries_with_regions))
            & (df["RegionName"].isin(kept_regions))
        )
    ]

    df.reset_index(drop=True, inplace=True)

    return df


def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def update_repo(repo='leaf-ai/covid-xprize',target_folder='covid-xprize-uptodate', overwrite=True):
    if os.path.exists(f'./{target_folder}'):
        if overwrite:
            shutil.rmtree(f'./{target_folder}', onerror=onerror)
            print(f'Overwriting...')
        else:
            print(f'{target_folder} already exists and not overwriting. Done.')
            return
    else:
        print(f'Cloning...')
    # Clone repo
    if platform.system() == 'Windows':
        os.system(f'cmd /c "git clone https://github.com/{repo}.git {target_folder}"')
    else:
        os.system(f"git clone https://github.com/{repo}.git {target_folder}")
    # Delete .git dir and .gitignore to avoid errors in current repo
    shutil.rmtree(f'./{target_folder}/.git', onerror=onerror)
    os.remove(f'./{target_folder}/.gitignore')
    print(f'Successfully cloned to {target_folder}')