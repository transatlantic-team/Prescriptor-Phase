# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

"""
This is the prescribe.py script for a simple example prescriptor that
generates IP schedules that trade off between IP cost and cases.

The prescriptor is "blind" in that it does not consider any historical
data when making its prescriptions.

The prescriptor is "greedy" in that it starts with all IPs turned off,
and then iteratively turns on the unused IP that has the least cost.

Since each subsequent prescription is stricter, the resulting set
of prescriptions should produce a Pareto front that highlights the
trade-off space between total IP cost and cases.

Note this file has significant overlap with ../random/prescribe.py.
"""

import os
import argparse
import pathlib
import numpy as np
import pandas as pd
from trans_presc import utils

NUM_PRESCRIPTIONS = 10

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

IP_COLS = list(IP_MAX_VALUES.keys())


def prescribe(start_date_str: str,
              end_date_str: str,
              path_to_hist_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:

    # Load historical IPs, just to extract the geos
    # we need to prescribe for.
    hist_df = pd.read_csv(path_to_hist_file,
                          parse_dates=['Date'],
                          encoding="ISO-8859-1",
                          keep_default_na=False,
                          error_bad_lines=True)

    # Load the IP weights, so that we can use them
    # greedily for each geo.
    #weights_df = pd.read_csv(path_to_cost_file, keep_default_na=False)
    
    hist_data_file = presc_filename = os.path.join(pathlib.Path(__file__).parent.absolute(), 'OxCGRT_latest.csv')
    cases_df = utils.get_num_cases(hist_data_file, start_date_str)
    
    
    #presc_filename = os.path.join(pathlib.Path(__file__).parent.absolute(), 'presc_analysis.csv')
    presc_filename = os.path.join(pathlib.Path(__file__).parent.absolute(), 'presc_analysis_full.csv')
    w_presc = utils.get_weighted_prescriptions(presc_filename, path_to_cost_file)

    # Generate prescriptions
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    

    N = (end_date - start_date).days / 30
    prescription_df = []
    
    
    # Define prescription strategies        
    policies = utils.get_policies()
    selected_strategies = [
        'alice_7', # Prescriptor 0
        'xbaro_3', # Prescriptor 1        
        'mcepeda_2', # Prescriptor 2
        'xbaro2_3', # Prescriptor 3
        'xbaro2_2', # Prescriptor 4
        'xbaro_1', # Prescriptor 5
        'alice_0', # Prescriptor 6
        'xbaro_8', # Prescriptor 7
        'xbaro_5', # Prescriptor 8
        'xbaro_0', # Prescriptor 9
    ]

    # Apply prescription strategies
    prescription_idx = 0
    for strategy in selected_strategies: 

        policy = policies[strategy]

        if policy['method'] == 'cases':
            # Apply prescription based on number of cases
            presc = utils.apply_policy_by_cases3(w_presc, cases_df, start_date, end_date, 
                                                 min_cases_w=policy['min_cases_w'], 
                                                 cost_w=policy['cost_w'])
        else:
            # Apply prescription based on weighting strategy
            step = policy['step']
            if isinstance(step, str):
                step = eval(step)
            presc = utils.apply_policy(w_presc, start_date, end_date, 
                                       method=policy['method'], 
                                       step=step, 
                                       initial_weights=policy['initial_weights'])

        presc['PrescriptionIndex'] = prescription_idx
        prescription_df.append(presc)
        prescription_idx += 1
        
        
    # Create dataframe from dictionary.
    prescription_df = pd.concat(prescription_df)

    # Create the directory for writing the output file, if necessary.
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save output csv file.
    prescription_df.to_csv(output_file_path, index=False)

    return


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
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prev_file, args.cost_file, args.output_file)
    print("Done!")
