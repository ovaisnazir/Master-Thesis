# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.timer import timer
import datetime as dt
import re
import os

def reduce_mem_usage(df):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
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


def import_data(file):
    """Creates a dataframe and optimize its memory usage."""
    df = pd.read_csv(file, keep_date_col=True)
    df = reduce_mem_usage(df)  
                                                          
    return df


def add_new_cols(new_cols, df):
    """Adds new colums to a data frame."""
    for col in new_cols:
        df[col] = np.nan   


def split_data_by_date(date, X, y):
    """
    It splits X and y sets by a 'Time' value 
    into sets for training and testing. 
        - Return: a dictionary with the four sets
                  (X_train, y_train, X_test, y_test)
    """
    sets = {}
    date_cut = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    
    X_test = X[X['Time'] > date_cut]
    X_train = X[X['Time'] <= date_cut]
    y_train = y[y.ID.isin(X_train.ID)]
    y_test = y[y.ID.isin(X_test.ID)]
    
    sets['X_train'] = X_train
    sets['X_test'] = X_test
    sets['y_train'] = y_train
    sets['y_test'] = y_test
    
    return sets


def input_missing_values(df, cols):
    """Imputes missing values based on the gap time between forecasted timestamp and NWP run."""
    regex = 'NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_(?P<weather_var>\w{1,4})'
    p = re.compile(regex)  
    
    NWP_met_vars_dict = {
        '1': ['U','V','T'],
        '2': ['U','V'],
        '3': ['U','V','T'],
        '4': ['U','V','CLCT']
    }
    
    for col in reversed(cols):
        m = p.match(col)
        col_name = 'NWP' + m.group('NWP') + '_' +  m.group('run') + '_' + m.group('fc_day') + '_' + m.group('weather_var')

        for key, value in NWP_met_vars_dict.items():
            for i in value:
                if m.group('NWP') == key and m.group('weather_var') == i:
                    df['NWP'+ key + '_' + i] = df['NWP'+ key + '_' + i].fillna(df[col_name])
    
    return df


def interpolate_missing_values(df, cols, index):
    """Imputes those missing values due to the NWP's frequency in data providing."""
    df.index = df[index]
    del df[index]
    
    for var in cols:
        df[var].interpolate(method='time', inplace=True, limit=2, limit_direction='both')
    df.reset_index(inplace=True)
    
    return df


def export_data(folder, WF, X_train, X_test, y_train, y_test):
    """Export data frames to specified folder."""
    os.makedirs(folder + '/' + WF, exist_ok=True)
    X_train.to_csv(folder + '{}/X_train.csv'.format(WF), index=False)
    X_test.to_csv(folder + '{}/X_test.csv'.format(WF), index=False)
    y_train.to_csv(folder + '{}/y_train.csv'.format(WF), index=False)
    y_test.to_csv(folder + '{}/y_test.csv'.format(WF), index=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    It creates data sets X_train, X_test, y_train, y_test from raw data (data/raw) 
    for every Wind Farm (data/interim) by:
        - Imputing missing values.
        - Selecting the best weather predictors by NWP.
        - Dropping non used attributes in next steps of the ML flow.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Import raw data
    with timer("Loading raw data"):
        X_train = import_data(input_filepath + 'X_train_raw.csv')
        X_train['Time'] = pd.to_datetime(X_train['Time'], format='%d/%m/%Y %H:%M')
        y_train = import_data(input_filepath + 'y_train_raw.csv')
        
        
    # Split raw data into train and test df's
    with timer("Generating train and test cleaned data sets by WF"):
        train_test_dfs = split_data_by_date('2018-11-13 23:00:00', X_train, y_train)
        X_train_2 = train_test_dfs.get('X_train')
        X_test = train_test_dfs.get('X_test')
        y_train_2 = train_test_dfs.get('y_train')
        y_test = train_test_dfs.get('y_test')
        
        # Wind farm list
        WF_lst = X_train['WF'].unique()
        
        # New columns that will be added to data sets
        new_cols = ['NWP1_U','NWP1_V','NWP1_T','NWP2_U','NWP2_V','NWP3_U',
                'NWP3_V','NWP3_T','NWP4_U','NWP4_V','NWP4_CLCT']
        
        # Columns that need interpolating missing values
        col_list = ['NWP2_U','NWP2_V','NWP3_U','NWP3_V','NWP3_T']
        
        for WF in WF_lst:
            # Make a copy of the data to not losing its initial format
            X_train_cpy = X_train_2.copy()
            X_test_cpy  = X_test.copy()
            y_train_cpy = y_train_2.copy()
            y_test_cpy = y_test.copy()
                
            # Row selection by WF
            X_train_cpy = X_train_cpy[X_train_cpy['WF'] == WF]
            X_test_cpy = X_test_cpy[X_test_cpy['WF'] == WF]
            
            # Save observations identification
            ID_train = X_train_cpy['ID']
            ID_test = X_test_cpy['ID']
        
            # Row selection for y_train
            y_train_cpy = y_train_cpy[['ID','Production']]
            y_train_cpy = y_train_cpy[y_train_cpy.ID.isin(ID_train.values)]
            
            # Row selection for y_test
            y_test_cpy = y_test_cpy[['ID','Production']]
            y_test_cpy = y_test_cpy[y_test_cpy.ID.isin(ID_test.values)]
        
            # Add new columns to X_train and X_test
            add_new_cols(new_cols, X_train_cpy)
            add_new_cols(new_cols, X_test_cpy)
            
            # Impute missing values
            X_train_cpy = input_missing_values(X_train_cpy, X_train.columns[3:])
            X_test_cpy = input_missing_values(X_test_cpy, X_test.columns[3:-9])
            interpolate_missing_values(X_train_cpy, col_list, 'Time')
            interpolate_missing_values(X_test_cpy, col_list, 'Time')   
     
            # Select the best NWP predictions for weather predictors
            X_train_cpy['U'] = X_train_cpy.NWP1_U
            X_train_cpy['V'] = X_train_cpy.NWP1_V
            X_train_cpy['T'] = X_train_cpy.NWP3_T
            X_train_cpy['CLCT'] = X_train_cpy.NWP4_CLCT
            
            X_test_cpy['U'] = X_test_cpy.NWP1_U
            X_test_cpy['V'] = X_test_cpy.NWP1_V
            X_test_cpy['T'] = X_test_cpy.NWP3_T
            X_test_cpy['CLCT'] = X_test_cpy.NWP4_CLCT
         
            X_train_cpy = X_train_cpy[['ID','Time','U','V','T','CLCT']]
            X_test_cpy = X_test_cpy[['ID','Time','U','V','T','CLCT']]
            
            # Export data to csv
            export_data(output_filepath, WF, X_train_cpy, X_test_cpy, y_train_cpy, y_test_cpy)
            
            
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
