import os
import urllib.request

import pandas as pd

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
