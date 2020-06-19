# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.timer import timer


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
    """Create a dataframe and optimize its memory usage."""
    df = pd.read_csv(file, keep_date_col=True)
    df = reduce_mem_usage(df)                                                            
    return df


def get_df_for_eda(df, regex = r'NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_'):
    """Convert the dataframe (test/train) to a convenient format for EDA, without changing the data itself."""
    # Create a temporal dataframe
    df_tmp = pd.DataFrame([])  

    # Regular expresion to capture the values from the column names
    p = re.compile(regex)
    
    # Get prefix list
    cols = df.columns[3:-1] 

    for col in cols:
        
        # Create a second temporal dataframe
        df_tmp2 = pd.DataFrame(np.nan, index=df.index, columns=[
            'WF','NWP','fc_day','run','id','time','U','V','T','CLCT','production'   
        ])  
        
        # Get values using the regex
        m = p.match(col)
        prefix = 'NWP' + m.group('NWP') + '_' + m.group('run') + '_' + m.group('fc_day') + '_'
               
        # Populate
        df_tmp2['WF'] = df['WF']
        df_tmp2['NWP'] = m.group('NWP')
        df_tmp2['fc_day'] = m.group('fc_day')
        df_tmp2['run'] = m.group('run')
        df_tmp2['id'] = df['ID']
        df_tmp2['time'] = df['Time']
        
        # Some of these weather parameters may not exist for every column
        try:
            df_tmp2['U'] =  df[prefix + 'U']
        except KeyError:
            pass
        
        try:
            df_tmp2['V'] = df[prefix + 'V']
        except KeyError:
            pass
        
        try:
            df_tmp2['T'] = df[prefix + 'T']
        except KeyError:
            pass
        
        try:
            df_tmp2['CLCT'] = df[prefix + 'CLCT']
        except KeyError:
            pass
        
        # Just in case there's not 'Production' column (f.i., X_test)
        if 'Production' in df.columns:
            df_tmp2['production'] = df['Production']
        else: 
            df_tmp2['production'] = np.nan
 
        df_tmp = df_tmp.append(df_tmp2, ignore_index=True) 
        del df_tmp2
        
    return df_tmp


def get_data_by_wf(df, folder):
    """Filter the data by WF."""
    df_lst = []
    wf_lst = df['WF'].unique()
    
    for wf in wf_lst:
        df_lst.append(df[df['WF'] == wf])
    
    for i in range(len(df_lst)):
        del df_lst[i]['WF']
        df_lst[i].to_csv(folder + '/for_EDA_by_WF/df_WF{}.csv'.format(i+1), index=False)
        
    return df_lst


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        data ready for exploratory data analysis (saved in ../interim/for_EDA_by_WF).
    """
    logger = logging.getLogger(__name__)
    logger.info('making EDA data set from raw data')
    
    # import raw data
    with timer("Loading raw data"):
        X_train = import_data(input_filepath + 'X_train_raw.csv')
        X_train['Time'] = pd.to_datetime(X_train['Time'])

        
    with timer("Formatting data for EDA"):
        # EDA df with all WF
        eda_df = get_df_for_eda(X_train)
        
        # Splitted by WF
        get_data_by_wf(eda_df, output_filepath)
        
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()