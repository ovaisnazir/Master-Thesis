# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import re
from collections import OrderedDict
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        data ready for exploratory data analysis (saved in ../interim/for_EDA_by_WF).
    """
    logger = logging.getLogger(__name__)
    logger.info('making EDA data set from raw data')
    
    # importa raw data
    import_data(input_filepath)

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


def import_data(file):
    """Create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, keep_date_col=True)
    df = reduce_mem_usage(df)                                                            
    return df


def walklevel(some_dir, level):
    """
       It works just like os.walk, but you can pass it a level parameter
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


def load_csv_files(loc, level=0):
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
    

def get_col_prefixes(cols, regex):
    
    prefix_lst = [] 
    p = re.compile(regex)
    
    for col in cols:
        m = p.match(col)
        
        if m is not None:
            col_prefix = 'NWP' + m.group('NWP') + '_' + m.group('met_var')
            prefix_lst.append(col_prefix)
    
    prefix_lst = list(OrderedDict.fromkeys(prefix_lst))
        
    return prefix_lst


def get_df_for_eda(df, regex = r'NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_'):
    """
        Convert the dataframe (test/train) to a convenient format for EDA,
        without changing the data itself.      
    """
    # Create a temporal dataframe
    df_tmp = pd.DataFrame([])  

    # Regular expresion to capture the values from the column names
    p = re.compile(regex)
    
    # Get prefix list
    cols = df.columns[3:-1] 
    prefix_lst = get_col_prefixes(cols, regex)
    
    for prefix in prefix_lst:
        
        # Create a second temporal dataframe
        df_tmp2 = pd.DataFrame(np.nan, index=df.index, columns=[
            'WF','NWP','fc_day','run','id_target','time','U','V','T','CLCT','production'   
        ])  
        
        # Get values using the regex
        m = p.match(prefix)
        
        # Select df columns that start with col_prefix
        sub_df = df.filter(regex='^' + prefix, axis=1)
        
        # Populate
        df_tmp2['WF'] = df['WF']
        df_tmp2['NWP'] = m.group('NWP')
        df_tmp2['FC_Day'] = m.group('fc_day')
        df_tmp2['Run'] = m.group('run')
        df_tmp2['ID_target'] = df['ID']
        df_tmp2['Time'] = df['Time']
        
        # Some of these weather parameters may no exist for every column
        try:
            df_tmp2['U'] = sub_df[prefix + 'U']
        except KeyError:
            pass
        
        try:
            df_tmp2['V'] = sub_df[prefix + 'V']
        except KeyError:
            pass
        
        try:
            df_tmp2['T'] = sub_df[prefix + 'T']
        except KeyError:
            pass
        
        try:
            df_tmp2['CLCT'] = sub_df[prefix + 'CLCT']
        except KeyError:
            pass
        
        # Just in case there's not 'Production' column (f.i., X_test)
        if 'Production' in df.columns:
            df_tmp2['Production'] = df['Production']
        else: 
            df_tmp2['Production'] = np.nan
 
        df_tmp = df_tmp.append(df_tmp2, ignore_index=True) 
        del df_tmp2
        
    return df_tmp


def get_data_by_wf(df, folder):
    df_lst = []
    wf_lst = df['WF'].unique()
    
    for wf in wf_lst:
        df_lst.append(df[df['WF'] == wf])
    
    for i in range(len(df_lst)):
        del df_lst[i]['WF']
        df_lst[i].to_csv(folder + '/df_WF{}.csv'.format(i+1), index=False)
        
    return df_lst


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()