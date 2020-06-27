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
from metpy import calc
from metpy.units import units
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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


# lambda function to get wind velocity module
get_wind_velmod = lambda x : float(calc.wind_speed(
    x.U * units.meter/units.second, 
    x.V * units.meter/units.second
).magnitude)



def fix_negative_clct(df):
    """Replaces negative values of CLCT by 0."""
    df.loc[df['CLCT'] < 0, 'CLCT'] = 0.0
    

def save_fig(fig_id, folder, WF, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirs(folder + '/' + WF, exist_ok=True)
    path = os.path.join(folder + '/' + WF, fig_id + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()
        
    plt.savefig(path, format=fig_extension, dpi=resolution)


def find_outliers(X, y, fig_id, folder, wf):
    """Finds outliers based on power curve, using DBSCAN algorithm."""
    
    # Add 'Production' column
    X['Production'] = list(y['Production'])

    # Calculate wind velocity module
    X['vel'] = X.apply(get_wind_velmod, axis=1)

    # Build data matrix
    X1 = X['vel'].values.reshape(-1,1)
    X2 = X['Production'].values.reshape(-1,1)
    X = np.concatenate((X1,X2), axis=1)

    # Select appropiate values for DBSCAN hyperparameters
    # Interim, I need to find a better way to automate the outlier detection.
    if wf == 'WF1':
        eps = 0.3
        min_samples = 55 
    elif wf == "WF2":
        eps = 0.47
        min_samples = 30
    elif wf == "WF3":
        eps = 0.7
        min_samples = 55
    elif wf == "WF4":
        eps = 0.5
        min_samples = 30
    elif wf == "WF5":
        eps = 0.25
        min_samples = 8
    else:
        eps = 0.18
        min_samples = 20

    # Using DBSCAN to find outliers
    outlier_detection = DBSCAN(eps = eps, 
                               metric = "mahalanobis", 
                               algorithm = 'brute', 
                               min_samples = min_samples, 
                               n_jobs = -1)
    clusters = outlier_detection.fit_predict(X)
    outliers = np.where(clusters == -1)[0]
    
    # Plot outliers
    plt.scatter(*X.T, color='yellowgreen', marker='.', label='Inliers')
    plt.scatter(*X[outliers].T, color='gold', marker='.', label='Outliers')
    plt.legend(loc='lower right')
    plt.xlabel("wind speed [m/s]")
    plt.ylabel("wind power [MWh]")
    save_fig(fig_id, folder, wf)

    return outliers



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
@click.argument('fig_output_filepath', type=click.Path())

def main(input_filepath, output_filepath, fig_output_filepath):
    """ 
    It creates data sets X_train, X_test, y_train, y_test from raw data (data/raw) 
    for every Wind Farm (data/interim) by:
        - Imputing missing values.
        - Selecting the best weather predictors by NWP.
        - Dropping non used attributes in next steps of the ML flow.
        - Handling outliers and abnormal data
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Import raw data
    with timer("Loading raw data"):
        X = import_data(input_filepath + 'X_train_raw.csv')
        X['Time'] = pd.to_datetime(X['Time'], format='%d/%m/%Y %H:%M')
        y = import_data(input_filepath + 'y_train_raw.csv')
        

    with timer("Making data sets by Wind Farm"):
        # Wind farm list
        WF_lst = X['WF'].unique()
        
        # New columns that will be added to data sets
        new_cols = ['NWP1_U','NWP1_V','NWP1_T','NWP2_U','NWP2_V','NWP3_U',
                'NWP3_V','NWP3_T','NWP4_U','NWP4_V','NWP4_CLCT']
        
        # Columns that need interpolating missing values
        col_list = ['NWP2_U','NWP2_V','NWP3_U','NWP3_V','NWP3_T']
        
        for WF in WF_lst:
            # Make a copy of the data to not losing its initial format
            X_cpy = X.copy()
            y_cpy = y.copy()
                
            # Row selection by WF
            X_cpy = X_cpy[X_cpy['WF'] == WF]
            
            # Save observations identification
            ID_X = X_cpy['ID']
        
            # Row selection for y
            y_cpy = y_cpy[['ID','Production']]
            y_cpy = y_cpy[y_cpy.ID.isin(ID_X.values)]
            
            # Add new columns to X
            add_new_cols(new_cols, X_cpy)
            
            # Impute missing values
            X_cpy = input_missing_values(X_cpy, X.columns[3:])
            interpolate_missing_values(X_cpy, col_list, 'Time')  
     
            # Select the best NWP predictions for weather predictors
            X_cpy['U'] = X_cpy.NWP1_U
            X_cpy['V'] = X_cpy.NWP1_V
            X_cpy['T'] = X_cpy.NWP3_T
            X_cpy['CLCT'] = X_cpy.NWP4_CLCT
            
            # Select final features
            X_cpy = X_cpy[['ID','Time','U','V','T','CLCT']]
    
            # Detect outliers and abnormal data

            fix_negative_clct(X_cpy)
            outliers = find_outliers(X_cpy, y_cpy, "outliers", fig_output_filepath, WF)
            plt.close()
            X_cpy.drop(X_cpy.index[list(outliers)], inplace=True)
            y_cpy.drop(y_cpy.index[list(outliers)], inplace=True)
                
            
            # Split data into X_train, X_test, y_train, y_test
            with timer("Spliting data for {} into train and test df's".format(WF)):
                train_test_dfs = split_data_by_date('2018-11-13 23:00:00', X_cpy, y_cpy)
                X_train = train_test_dfs.get('X_train')
                X_test = train_test_dfs.get('X_test')
                y_train = train_test_dfs.get('y_train')
                y_test = train_test_dfs.get('y_test')
                
                # Export data to csv
                export_data(output_filepath, WF, X_train, X_test, y_train, y_test)
    
                  
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
