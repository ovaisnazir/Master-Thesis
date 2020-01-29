import os
import pandas as pd
import numpy as np
import datetime as dt
import gc
from src.functions import data_import as dimp
from src.functions import data_exploration as dexp
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas_profiling


def get_report_by_site(df, sites):
    """Generate a report for every site in list 'sites'
    """
    reports = {}
    for site in sites:
        try:
            report = df.loc[df['site_id'] == site, :].profile_report(
                title='Report Site {}'.format(site), 
                style={'full_width':True}
            )          
            reports[site] = report
        except Exception:
            print('WARN: Report for site {} has not been generated'.format(site))
            continue
                    
    return reports


def export_reports(df, reports, loc):
    """ Export each report in 'reports' to html in the location indicated by 'loc'
    """
    for key in reports.keys():
        try:
            reports[key].to_file(
                output_file = loc + '{}_site{}.html'.format(df.name, key)      
            )
        except Exception:
            print('WARN: Exportation failed for site {}'.format(key))
            continue
            
def test_interpolate_methods(df, col, methods, n, order):
    '''
      Function to test several methods of the interpolate() pandas method
      for imputing NaN/NULL values. Inputs:
          - df: data frame with no missing values in the column we want to test.
          - col: string with the name of the variable.
          - methods: list of interpolate methods to test.
          - n: number of NaNs to be randomly imputed in the original data ts.
          - order: order for spline/polynomial methods.
    '''
    df_cpy = df.copy()
    
    # Random selection of the time indexes 
    nan_indexes = df_cpy.sample(n, replace=False, random_state=1).index
    
    # Impute nan's in a copy of the original df
    df_cpy.loc[nan_indexes, col] = np.nan
    
    for m in methods:
        if m in ['spline','polynomial']:
            df_cpy[col].interpolate(method=m, order=order, inplace=True)         
        else:
            # Interpolation
            df_cpy[col].interpolate(method=m, inplace=True)
            df_cpy2 = df_cpy.copy()

            # Error calculation
            aprox = df_cpy.loc[list(nan_indexes), col]
            real = df.loc[list(nan_indexes), col]
            error_vec = (abs(aprox - real)/real) * 100
            err_mean = np.mean(error_vec)
            err_median = np.median(error_vec)

            # Set back NaN values for the next method
            df_cpy.loc[nan_indexes, col] = np.nan
            
        print('Method {}:'.format(m))
        print('Error vector:', error_vec)
        print('Mean error: {0:.2f}'.format(err_mean))
        print('Median error: {0:.2f}'.format(err_median))
        print('===================================')
    
    return df_cpy2

def show_nan(df):
    """ Creates a dictionary with:
            - key: name of each column of df
            - value: index/es of df column with NaN/Null value
    """
    results = {}
    for col in df.columns:
        results[col] = df[col][df[col].isna()]               
    return results

def get_missing_percentage(df, *index_level):
    """
        Returns the percentage of missing values in every column
        of a dataframe, optionally grouping by an index_level .
    """
    f = lambda x : round((x.isna().sum() / x.shape[0]) * 100, 2)
    if index_level:
        nans = df.groupby(level=index_level).apply(f)
    else:
        nans = df.apply(f)
    return nans

