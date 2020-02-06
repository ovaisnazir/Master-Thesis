import os
import pandas as pd
import numpy as np
import datetime as dt
import gc

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file, *date_col):
    """create a dataframe and optimize its memory usage"""
    if date_col:
        df = pd.read_csv(file, parse_dates=[date_col])
        df = reduce_mem_usage(df)
    else:
        df = pd.read_csv(file, keep_date_col=True)
        df = reduce_mem_usage(df)
                                                               
    return df


def walklevel(some_dir, level):
    """It works just like os.walk, but you can pass it a level parameter
       that indicates how deep the recursion will go.
       If depth is -1 (or less than 0), the full depth is walked.
    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def load_data(loc, level=0):
    """ It loads all csv files in a location and returns a list of data frames 
        created from those files. Inputs:
            - loc: directory where the csv files are located.
            - level: how deep the recursion will go in the directory. By default is 0.
    """
    df_list = []
    for dirname, _, filenames in walklevel(loc, level):
        for filename in filenames:
            print('**' + filename + ':')
            df_list.append(import_data(dirname + filename))
            print('\n')
    
    return df_list

def import_raw_data(loc):
    df_list = load_data(loc) 
    df = pd.merge(
        pd.merge(df_list[0], df_list[1], how='left', on='building_id'),
        df_list[2],
        how='left',
        on=['site_id','timestamp']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])   
    df['year_built'] = pd.array(df['year_built'], dtype=pd.Int16Dtype())
    df['floor_count'] = pd.array(df['floor_count'], dtype=pd.Int16Dtype())

    # Rearranging columns
    cols = [
        'site_id',
        'building_id', 
        'year_built', 
        'primary_use', 
        'floor_count', 
        'square_feet',
        'meter',  
        'timestamp',
        'air_temperature', 
        'cloud_coverage',
        'dew_temperature',
        'precip_depth_1_hr',
        'sea_level_pressure',
        'wind_direction',
        'wind_speed',
        'meter_reading'
    ]

    df = df[cols]
    return df

def get_data_by_site(df, site):
    df = df[df['site_id'] == site]
    del df['site_id']
    return df
