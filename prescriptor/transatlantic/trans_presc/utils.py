# Copyright 2020 (c) Transatlantic XPrize Team

"""
This file contains useful methods for prescribtion
"""

import os
import math
import numpy as np
import pandas as pd
#from covid_xprize.scoring.prescriptor_scoring import weight_prescriptions_by_cost
from .xprize import weight_prescriptions_by_cost


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

NPI_COLUMNS = list(IP_MAX_VALUES.keys())


def add_geo_id(df):
    
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])
    return df


def get_num_cases(hist_data_file, date):    
    hist_cases_df = pd.read_csv(hist_data_file,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    hist_cases_df = add_geo_id(hist_cases_df)
    
    # Add new cases column
    hist_cases_df['NewCases'] = hist_cases_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
    # Fill any missing case values by interpolation and setting NaNs to 0
    hist_cases_df.update(hist_cases_df.groupby('GeoID').NewCases.apply(
        lambda group: group.interpolate()).fillna(0))
    
    # Check if the provided date is in the historical data
    if isinstance(date, str):
        date = pd.to_datetime(date, format='%Y-%m-%d')
    if date not in hist_cases_df.Date.to_list() or date == hist_cases_df.Date.max():        
        # If given date is not in historical data, use the last available date
        date = hist_cases_df.Date.max() - pd.Timedelta(days=1)
        
    return hist_cases_df[hist_cases_df['Date'] == date][['CountryName', 'RegionName', 'GeoID', 'NewCases']]


def get_weighted_prescriptions(presc_file, cost_df):
    final_prescriptors = pd.read_csv(presc_file,                  
                                     encoding="ISO-8859-1",                  
                                     error_bad_lines=True)
    if isinstance(cost_df, str):
        cost_df = pd.read_csv(cost_df)

    start_date = "2020-01-01"
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')

    p_date = start_date_val
    dated_prescriptions = final_prescriptors
    dated_prescriptions['Date'] = (pd.date_range(start=start_date_val, periods=dated_prescriptions.shape[0], freq='D'))

    region_prescriptions = []
    for _, row in cost_df.iterrows():
        reg_presc = dated_prescriptions.copy()
        reg_presc['CountryName'] = row['CountryName']
        reg_presc['RegionName'] = row['RegionName']
        region_prescriptions.append(reg_presc)

    region_prescriptions = pd.concat(region_prescriptions)

    w_presc = weight_prescriptions_by_cost(region_prescriptions, cost_df)
    
    w_presc['WCost'] = w_presc[NPI_COLUMNS].sum(axis=1)

    w_presc = w_presc[['CountryName', 'RegionName', 'PrescriptorName', 'NormCost', 'WCost']]
    
    top_w = w_presc[(w_presc['NormCost'] < 0.0000001)]
    max_vals = top_w.rename(columns = {'WCost': 'MaxWCost'}, inplace = False)[['CountryName', 'RegionName', 'MaxWCost']]
    
    presc_set = pd.merge(w_presc, max_vals, on=['CountryName', 'RegionName'], how='left')
    
    presc_set['NormWCost'] = presc_set['WCost'] / presc_set['MaxWCost']    
    
    presc_set[NPI_COLUMNS] = presc_set['PrescriptorName'].str.split('_', expand=True).apply(lambda x: x.str.slice(0, 1)).astype('int32')
    
    return presc_set[['CountryName', 'RegionName', 'PrescriptorName', 'NormCost', 'NormWCost'] + NPI_COLUMNS]


def filter_prescriptions_by_relevance(weighted_prescriptions, cases_w, cost_w):
    
    weighted_prescriptions['W1'] = weighted_prescriptions['NormCost'] * cases_w
    weighted_prescriptions['W2'] = weighted_prescriptions['NormWCost'] * cost_w
    
    weighted_prescriptions['Relevance'] = 1 - ((weighted_prescriptions['W1'] + weighted_prescriptions['W2']) / 2)
    
    weighted_prescriptions = add_geo_id(weighted_prescriptions)

    weighted_prescriptions['r'] = weighted_prescriptions.groupby('GeoID')['Relevance'].rank('dense', ascending=False)
       
    #weighted_prescriptions = weighted_prescriptions[weighted_prescriptions['r'] == 1.0]
    weighted_prescriptions = weighted_prescriptions.sort_values(['GeoID', 'r'])
    weighted_prescriptions = weighted_prescriptions.groupby('GeoID').first()

    
    return weighted_prescriptions[['CountryName', 'RegionName', 'PrescriptorName', 'NormCost', 'NormWCost'] + NPI_COLUMNS]


def filter_prescriptions_by_relevance2(weighted_prescriptions, cost_w):
    
    weighted_prescriptions['W1'] = weighted_prescriptions['NormCost'] *  weighted_prescriptions['W']
    weighted_prescriptions['W2'] = weighted_prescriptions['NormWCost'] * cost_w
    
    weighted_prescriptions['Relevance'] = 1 - ((weighted_prescriptions['W1'] + weighted_prescriptions['W2']) / 2)
        
    weighted_prescriptions = add_geo_id(weighted_prescriptions)

    weighted_prescriptions['r'] = weighted_prescriptions.groupby('GeoID')['Relevance'].rank('dense', ascending=False)
       
    #weighted_prescriptions = weighted_prescriptions[weighted_prescriptions['r'] == 1.0]
    weighted_prescriptions = weighted_prescriptions.sort_values(['GeoID', 'r'])
    weighted_prescriptions = weighted_prescriptions.groupby('GeoID').first()

    
    return weighted_prescriptions[['CountryName', 'RegionName', 'PrescriptorName', 'NormCost', 'NormWCost'] + NPI_COLUMNS]


def update_weights(cases_w, cost_w, method, step, iteration=0):
    if method == 'constant':
        # Weights are not changed
        pass
    elif method == 'linear':
        # Apply step to weights
        cases_w = max(0.0, cases_w - step)
        cost_w = min(1.0, cost_w + step)
    elif method == 'linear-inv':
        # Apply step to weights
        cases_w = min(0.1, cases_w + step)
        cost_w = max(0.0, cost_w - step)
    elif method == 'exponential':
        # Update on exponential manner   
        factor = step * iteration        
        cases_w = max(0.0, cases_w - factor)
        cost_w = min(1.0, cost_w + factor)
    elif method == 'exponential-inv':
        # Update on exponential manner   
        factor = step * iteration        
        cases_w = min(1.0, cases_w + factor)
        cost_w = max(0.0, cost_w - factor)
    elif method == 'linear-double':
        # Update on exponential manner   
        step_cases, step_cost = step        
        cases_w = max(0.0, cases_w - step_cases)
        cost_w = min(1.0, cost_w + step_cost)
    elif method == 'log':
        #Update in a logarithmic manner
        factor = step * (1 / (iteration+1))
        cases_w = max(0.0, cases_w - factor)
        cost_w = min(1.0, cost_w + factor)
    elif method == 'log-expo':
        #Update in a logarithmic manner for cases and expo for cost
        cases_factor = step * (1 / (iteration+1))
        cost_factor = step * (iteration + 1)
        cases_w = max(0.0, cases_w - cases_factor)
        cost_w = min(1.0, cost_w + cost_factor)
    elif method == 'expo-log':
        #Update in a logarithmic manner for cost and expo for cases
        cost_factor = step * (1 / (iteration+1))
        cases_factor = step * (iteration + 1)
        cases_w = max(0.0, cases_w - cases_factor)
        cost_w = min(1.0, cost_w + cost_factor)
        
    return cases_w, cost_w


def apply_policy(weighted_prescriptions, start_date, end_date, step=0.01, initial_weights=[1.0, 0.0], method='constant'):
    
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_val = pd.to_datetime(end_date, format='%Y-%m-%d')

    prescriptions = []
    cases_w = initial_weights[0]
    cost_w = initial_weights[1]
    iteration = 0
    for prescription_date in pd.date_range(start=start_date_val, end=end_date_val, freq='D'):
        daily_presc = filter_prescriptions_by_relevance(weighted_prescriptions, cases_w, cost_w)
        daily_presc['Date'] = prescription_date
        prescriptions.append(daily_presc)
        cases_w, cost_w = update_weights(cases_w, cost_w, method, step, iteration)
        iteration += 1
    
    prescriptions = pd.concat(prescriptions)
    
    return prescriptions[['CountryName', 'RegionName', 'Date'] + NPI_COLUMNS].reset_index(drop=True)        


def apply_policy_by_cases(weighted_prescriptions, cases_df, start_date, end_date, min_cases_w=0.0, cost_w=0.5):
    
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_val = pd.to_datetime(end_date, format='%Y-%m-%d')
    
    cases_df['W'] = np.maximum(min_cases_w, np.minimum(1, cases_df['NewCases'] / cases_df['NewCases'].mean()))
    
    weighted_prescriptions = add_geo_id(weighted_prescriptions)
    
    prescriptions = []
    iteration = 0
    for prescription_date in pd.date_range(start=start_date_val, end=end_date_val, freq='D'):
        daily_presc = []
        for region in cases_df.GeoID.unique():
            cases_w = float(cases_df[cases_df['GeoID'] == region]['W'])
            region_daily_presc = filter_prescriptions_by_relevance(
                weighted_prescriptions[weighted_prescriptions['GeoID'] == region].copy(), 
                cases_w, cost_w
            )
            daily_presc.append(region_daily_presc)
        
        daily_presc = pd.concat(daily_presc)
        daily_presc['Date'] = prescription_date
        prescriptions.append(daily_presc)        
        iteration += 1
    
    prescriptions = pd.concat(prescriptions)
    
    return prescriptions[['CountryName', 'RegionName', 'Date'] + NPI_COLUMNS].reset_index(drop=True)   


def apply_policy_by_cases2(weighted_prescriptions, cases_df, start_date, end_date, min_cases_w=0.0, cost_w=0.5):
    
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_val = pd.to_datetime(end_date, format='%Y-%m-%d')
    
    cases_df['W'] = np.maximum(min_cases_w, np.minimum(1, cases_df['NewCases'] / cases_df['NewCases'].mean()))
    
    weighted_prescriptions = add_geo_id(weighted_prescriptions)
        
    daily_presc = []
    for region in cases_df.GeoID.unique():
        cases_w = float(cases_df[cases_df['GeoID'] == region]['W'])        
        region_presc = filter_prescriptions_by_relevance(
            weighted_prescriptions[weighted_prescriptions['GeoID'] == region].copy(), 
            cases_w, cost_w
        )
        daily_presc.append(region_presc)        
        
    daily_presc = pd.concat(daily_presc)
        
    prescriptions = []
    for prescription_date in pd.date_range(start=start_date_val, end=end_date_val, freq='D'):
        
        daily_presc['Date'] = prescription_date
        prescriptions.append(daily_presc.copy())        
    
    prescriptions = pd.concat(prescriptions)
    
    return prescriptions[['CountryName', 'RegionName', 'Date'] + NPI_COLUMNS].reset_index(drop=True)   


def apply_policy_by_cases3(weighted_prescriptions, cases_df, start_date, end_date, min_cases_w=0.0, cost_w=0.5):
    
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_val = pd.to_datetime(end_date, format='%Y-%m-%d')
    
    cases_df['W'] = np.maximum(min_cases_w, np.minimum(1, cases_df['NewCases'] / cases_df['NewCases'].mean()))
    
    # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
    cases_df = add_geo_id(cases_df)
    weighted_prescriptions = add_geo_id(weighted_prescriptions)
    
    weighted_prescriptions = pd.merge(weighted_prescriptions, cases_df[['GeoID', 'W']], on='GeoID')
    
    daily_presc = filter_prescriptions_by_relevance2(weighted_prescriptions, cost_w)
                
    prescriptions = []
    for prescription_date in pd.date_range(start=start_date_val, end=end_date_val, freq='D'):
        
        daily_presc['Date'] = prescription_date
        prescriptions.append(daily_presc.copy())        
    
    prescriptions = pd.concat(prescriptions)
    
    return prescriptions[['CountryName', 'RegionName', 'Date'] + NPI_COLUMNS].reset_index(drop=True)


def plot_weights_evolution(start_date, end_date, step=0.01, initial_weights=[1.0, 0.0], method='constant', ax=None, title=None):
    start_date_val = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_val = pd.to_datetime(end_date, format='%Y-%m-%d')
    cases_w = initial_weights[0]
    cost_w = initial_weights[1]

    results = []
    iteration = 0
    for prescription_date in pd.date_range(start=start_date_val, end=end_date_val, freq='D'):
        results.append(pd.DataFrame(data = {'Date': [prescription_date], 'CasesW': [cases_w], 'CostW': [cost_w]}))
        cases_w, cost_w = update_weights(cases_w, cost_w, method, step, iteration)
        iteration += 1

    results = pd.concat(results).set_index('Date')
    if title is None:
        title = method
    if ax is not None:
        results.plot(title=title, ax=ax)
    else:
        results.plot(title=title)
        
def get_policies():
    return {
        'alice_0': {
            'method': 'log',
            'step': 0.01,
            'initial_weights': [0.8, 0.2]
        },
        'alice_1': {
            'method': 'log',
            'step': 0.05,
            'initial_weights': [0.8, 0.2]
        },
        'alice_2': {
            'method': 'log',
            'step': 0.01,
            'initial_weights': [0.2, 0.8]
        },
        'alice_3': {
            'method': 'log',
            'step': 0.01,
            'initial_weights': [0.0, 1.0]
        },
        'alice_4': {
            'method': 'log',
            'step': 0.01,
            'initial_weights': [0.5, 0.5]
        },
        'alice_5': {
            'method': 'log',
            'step': 0.01,
            'initial_weights': [1.0, 0.0]
        },            
        'alice_6': {
            'method': 'expo-log',
            'step': 0.01,
            'initial_weights': [0.8, 0.2]
        },
        'alice_7': {
            'method': 'expo-log',
            'step': 0.01,
            'initial_weights': [0.2, 0.8]
        },
        'alice_8': {
            'method': 'log-expo',
            'step': 0.01,
            'initial_weights': [0.8, 0.2]
        },
        'alice_9': {
            'method': 'log-expo',
            'step': 0.01,
            'initial_weights': [0.5, 0.5]
        },
        'mcepeda_0': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.75, 0.25]
        },
        'mcepeda_1': {
            'method': 'constant',
            'step': 0.05,
            'initial_weights': [0.5, 0.5]
        },
        'mcepeda_2': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.25, 0.75]
        },
        'mcepeda_3': {
            'method': 'linear-double',
            'step': [0.01, 0.02],
            'initial_weights': [0.9, 0.1]
        },
        'mcepeda_4': {
            'method': 'linear-double',
            'step': [0.01, 0.005],
            'initial_weights': [0.5, 0.5]
        },
        'mcepeda_5': {
            'method': 'linear-inv',
            'step': 0.01,
            'initial_weights': [0.2, 0.7]
        },            
        'mcepeda_6': {
            'method': 'linear-inv',
            'step': 0.01,
            'initial_weights': [0.5, 0.5]
        },
        'mcepeda_7': {
            'method': 'exponential',
            'step': 0.01,
            'initial_weights': [0.9, 0.1]
        },
        'mcepeda_8': {
            'method': 'exponential-inv',
            'step': 0.01,
            'initial_weights': [0.2, 0.8]
        },
        'mcepeda_9': {
            'method': 'exponential-inv',
            'step': 0.01,
            'initial_weights': [0.5, 0.5]
        },
        'xbaro_0': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.9, 0.1]
        },
        'xbaro_1': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.7, 0.3]
        },
        'xbaro_2': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.5, 0.5]
        },
        'xbaro_3': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [0.2, 0.8]
        },
        'xbaro_4': {
            'method': 'constant',
            'step': 0.01,
            'initial_weights': [1.0, 0.9]
        },
        'xbaro_5': {
            'method': 'linear',
            'step': '0.01/N',
            'initial_weights': [1.0, 0.0]
        },            
        'xbaro_6': {
            'method': 'exponential',
            'step': '0.04/pow(10, max(2, N))',
            'initial_weights': [1.0, 0.0]
        },
        'xbaro_7': {
            'method': 'linear',
            'step': '0.01/N*2',
            'initial_weights': [1.0, 0.0]
        },
        'xbaro_8': {
            'method': 'exponential',
            'step': '0.04/pow(10, max(2, N) - 0.5)',
            'initial_weights': [1.0, 0.0]
        },
        'xbaro2_0': {
            'method': 'cases',
            'min_cases_w': 0.1,
            'cost_w': 0.5
        },
        'xbaro2_1': {
            'method': 'cases',
            'min_cases_w': 0.5,
            'cost_w': 0.5
        },
        'xbaro2_2': {
            'method': 'cases',
            'min_cases_w': 0.7,
            'cost_w': 0.5
        },
        'xbaro2_3': {
            'method': 'cases',
            'min_cases_w': 0.2,
            'cost_w': 0.5
        },
        'xbaro2_4': {
            'method': 'cases',
            'min_cases_w': 0.1,
            'cost_w': 0.8
        }
    }
 