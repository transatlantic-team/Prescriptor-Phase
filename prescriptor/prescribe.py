import os
import argparse
import numpy as np
import pandas as pd

NUM_PRESCRIPTIONS = 3

NPI_COLUMNS = [
    'C1_School closing',
    'C2_Workplace closing',
    'C3_Cancel public events',
    'C4_Restrictions on gatherings',
    'C5_Close public transport',
    'C6_Stay at home requirements',
    'C7_Restrictions on internal movement',
    'C8_International travel controls',
    'H1_Public information campaigns',
    'H2_Testing policy',
    'H3_Contact tracing',
    'H6_Facial Coverings'
]

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}

OPT_PRESCRIPTION = [3, 3, 0, 4, 2, 0, 0, 4, 0, 3, 0, 0]


def square_wave(t: int, t_up: int, t_down: int, start_up: bool = True):
    """Returns a square wave in {0,1}: 

    :param t: length of signal

    :param t_up: time samples in 1

    :param t_down: time samples in 0 (===> period = t_up + t_down and duty cycle = t_up/t_down)

    :param start_up: whether to start in 1 or 0
    """
    wave = np.zeros(t)

    time_up = 0
    time_down = 0
    is_up = start_up
    for i in range(t):
        if is_up:
            wave[i] = 1
            time_up += 1
            if time_up == t_up:
                time_up = 0
                is_up = not is_up
        else:
            time_down += 1
            if time_down == t_down:
                time_down = 0
                is_up = not is_up

    return wave


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    # Create skeleton df with one row for each prescription
    # for each geo for each day
    hdf = pd.read_csv(path_to_hist_file,
                      parse_dates=['Date'],
                      encoding="ISO-8859-1",
                      dtype={"RegionName": str},
                      error_bad_lines=True)
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    prescription_idxs = []
    country_names = []
    region_names = []
    dates = []

    geoids = (hdf["CountryName"] + "__" +
              hdf["RegionName"].astype(str)).unique()

    # Load country cost-per-npi dataframe
    weights_df = pd.read_csv(path_to_cost_file, keep_default_na=False)

    for prescription_idx in range(NUM_PRESCRIPTIONS):
        for country_name in hdf['CountryName'].unique():
            cdf = hdf[hdf['CountryName'] == country_name]
            for region_name in cdf['RegionName'].unique():
                for date in pd.date_range(start_date, end_date):
                    prescription_idxs.append(prescription_idx)
                    country_names.append(country_name)
                    region_names.append(region_name)
                    dates.append(date.strftime("%Y-%m-%d"))

    prescription_df = pd.DataFrame({
        'PrescriptionIndex': prescription_idxs,
        'CountryName': country_names,
        'RegionName': region_names,
        'Date': dates}).fillna("")

    # Initialize NPI columns to then index-access them
    for npi_col in NPI_COLUMNS:
        prescription_df[npi_col] = 0

    # end_date - start_date
    n_days = len(set(dates))
    # Challenge number of regions
    n_geoids = len(geoids)

    # Bang-bang prescriptor
    for id_i, geoid in enumerate(geoids):
        country, region = geoid.split("__")
        # Fix nan regions
        region = (region if region != "nan" else "")
        # Load geoid cost
        weights_geoid = weights_df[(weights_df['CountryName'] == country) & (
            weights_df['RegionName'] == region)][NPI_COLUMNS].values[0]

        for i, npi_col in enumerate(NPI_COLUMNS):
            # Per-day cost of prescription
            s_i = weights_geoid[i] * OPT_PRESCRIPTION[i]
            # Compute times
            npi_uptime = 4
            npi_downtime = 3
            # Generate prescription signal
            wave_i = square_wave(n_days, npi_uptime,
                                 npi_downtime, start_up=True)
            # Assign strategy
            prescription_df.loc[(prescription_df['PrescriptionIndex'] == 2) & (prescription_df['CountryName'] == country) & (
                prescription_df['RegionName'] == region), npi_col] = OPT_PRESCRIPTION[i] * wave_i

    # Trivial prescriptions all 0 and all MAX:
    # Per each index: we have total_dates * total_geoids rows to fill
    num_rows = n_days * n_geoids
    for npi_col, max_value in sorted(IP_MAX_VALUES.items()):
        # Trivial strategy 1: all NPIs to max to minimize number of new cases
        prescription_df.loc[prescription_df['PrescriptionIndex']
                            == 0, npi_col] = max_value * np.ones(num_rows, dtype=int)
        # Trivial strategy 2: all NPIs to 0 to minimize cost
        prescription_df.loc[prescription_df['PrescriptionIndex'] == 1, npi_col] = np.zeros(
            num_rows, dtype=int)

    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save to a csv file
    prescription_df.to_csv(output_file_path, index=False)

    return


# XPRIZE API -- DO NOT TOUCH
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prev_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(
        f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file,
              args.cost_file, args.output_file)
    print("Done!")
