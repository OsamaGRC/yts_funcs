import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import warnings

# Calculates Antecedent Precipitation Index (API) from daily rainfall data 
def generate_api(data, date_column:str, k:float):
    '''
    This function creates Antecedent Precipitation Index (API) from daily rainfall data 
    data: can be a path in 'str' format to a csv file or a dataframe
    date_column: name of the date column (expected to be aggregated to days)
    
    Return: API datafaram of the same size and and indecies to the input rainfall
    '''

    if isinstance(data, pd.DataFrame):
        # If input is a DataFrame, directly use it
        print("Processing DataFrame...")
        df = data.copy()
    elif isinstance(data, str):
        # If input is a string, assume it's a file path
        if data.endswith('.csv'):
            print("Loading CSV file...")
            # Read the CSV into a DataFrame
            df = pd.read_csv(data)
        else:
            raise ValueError("Input file must be a CSV file.")
    else:
        raise TypeError("Input must be either a DataFrame or a CSV file path.")
        
    if date_column !=None:
        try:
            df.set_index(date_column, inplace=True)
            print(f"Index set to '{date_column}' successfully.")
            
        except KeyError:
            # If the column doesn't exist, raise a warning
            warnings.warn(f"Column '{date_column}' does not exist in the DataFrame. Index not set.")
        except Exception as e:
            # Catch any other unforeseen exceptions and raise a warning
            warnings.warn(f"An unexpected error occurred: {e}")

    # Create empty dataframe for API 
    df_api= df.copy()
    df_api.loc[:,:]= np.nan

    # Calculating API
    rf = df.to_numpy()
    api = np.zeros_like(rf)
    k = k

    api[0,:] =rf[0,:]
    for i in range(1,rf.shape[0]):
        for j in range(rf.shape[1]):
                api[i, j] = k*api[i-1, j] + rf[i, j]

    # Storing calcs in API dataframe
    df_api.loc[:,:]=api
    print("API calculated successfully")
    return df_api
    
  
# Create a list of shorter dates interval from a longer data range
def dt_segmenter(start: str = None, end: str = None, dt_days: int = None, max_sig: int = 7):
    '''
    Splits a time interval into evenly spaced datetime ranges.

    Only two of the following should be provided: `start`, `end`, or `dt_days`.

    Parameters:
        start (str): Start datetime in format '%Y-%m-%dT%H:%M:%SZ'
        end (str): End datetime in format '%Y-%m-%dT%H:%M:%SZ'
        dt_days (int): Total number of days in the desired range
        max_sig (int): Number of segments

    Returns:
        list of tuples: Each tuple contains (start_datetime, end_datetime)
    '''
    ranges = []

    # Ensure only 2 out of the 3 parameters are provided
    args = [start is not None, end is not None, dt_days is not None]
    if sum(args) != 2:
        raise ValueError("Exactly two of 'start', 'end', or 'dt_days' must be specified.")

    if start and dt_days:
        storm_ref = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        step = dt_days / max_sig
        for i in range(max_sig):
            range_start = storm_ref + timedelta(days=i * step)
            range_end = range_start + timedelta(days=step)
            ranges.append((range_start, range_end))

    elif end and dt_days:
        storm_ref = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        step = dt_days / max_sig
        for i in range(max_sig):
            range_start = storm_ref + timedelta(days=i * step)
            range_end = range_start + timedelta(days=step)
            ranges.append((range_start, range_end))

    elif start and end:
        storm_ref = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        storm_rng = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ")
        total_range = storm_rng - storm_ref
        step = np.ceil((total_range / max_sig).total_seconds()/(24*60*60))
        print(f"There are {step} intervals in the date range provided")
        for i in range(np.int32(step)):
            range_start = storm_ref + timedelta(days=i * max_sig)
            range_end = range_start + timedelta(days=max_sig)
            ranges.append((range_start, range_end))

    return sorted(ranges)

    
def set_frequency(df, freq='15T', time_col=None, fill_method=None):
    """
    Regularizes a time series to the specified frequency.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        freq (str): Target frequency (e.g., '15T', '1H', 'D', etc.).
        time_col (str): Optional name of datetime column if not already the index.
        fill_method (str): One of 'ffill', 'bfill', 'interpolate', or None.

    Returns:
        pd.DataFrame: DataFrame with uniform time steps at the specified frequency.
    """
    
    # If datetime column is specified, convert and set as index
    if time_col:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index must be datetime or provide a valid 'time_col'.")

    # Sort index
    #df = df.sort_index()

    # Infer frequency to check uniformity
    inferred_freq = pd.infer_freq(df.index[:-20])

    if inferred_freq == freq:
        df.index.freq = freq
        print(f"PASSED: Time series already uniform at '{freq}, no further process required'.")
    else:
        print(f"CHECK: Time series not uniform — resampling to '{freq}, resampling applied'.")
        df = df.resample(freq).asfreq()
        
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'interpolate':
            df = df.interpolate()
        elif fill_method is not None:
            raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'interpolate', or None.")
    
    return df

import pandas as pd

def tz_shift_resample(
    df,
    source_tz='UTC',
    target_tz='Australia/Sydney',
    shift_hours=-9,
    resample_freq='D',
    label='right',
    closed='right'
):
    """
    Converts timezones, shifts the datetime index, resamples the data, 
    and shifts the index back.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index
        source_tz (str): Original timezone (e.g., 'UTC')
        target_tz (str): Target timezone (e.g., 'Australia/Sydney')
        shift_hours (int): Hours to shift index before and after resampling
        resample_freq (str): Resample frequency (e.g., 'D', 'H', 'W')
        label (str): 'right' or 'left' label for resample
        closed (str): 'right' or 'left' side for interval inclusion

    Returns:
        pd.DataFrame: Resampled DataFrame with adjusted time index
    """
    # Make sure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("DataFrame index must be a datetime type.")

    # Localize if naive
    if df.index.tz is None:
        df.index = df.index.tz_localize(source_tz)
    else:
        df.index = df.index.tz_convert(source_tz)

    # Convert to target timezone
    df.index = df.index.tz_convert(target_tz)

    # Shift index
    df.index = df.index + pd.Timedelta(hours=shift_hours)

    # Resample
    df = df.resample(resample_freq, label=label, closed=closed).sum()

    # Shift index back
    df.index = df.index - pd.Timedelta(hours=shift_hours)

    return df
