import pandas as pd
import numpy as np

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
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

def make_ts_index(df, timeCol):
    """create a datetime index from timeCol column"""
    df_cpy = df.copy()
    df_cpy[timeCol] = pd.to_datetime(df_cpy[timeCol])
    df_cpy.index = df_cpy[timeCol]    
    return df_cpy

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

            # Set back the NaN's for the next method
            df_cpy.loc[nan_indexes, col] = np.nan
            
        print('Method {}:'.format(m))
        print('Error vector:', error_vec)
        print('Mean error: {0:.2f}'.format(err_mean))
        print('Median error: {0:.2f}'.format(err_median))
        print('===================================')
    
    return df_cpy2