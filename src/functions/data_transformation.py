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
import re
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from metpy import calc
from metpy.units import units

# Obtención del módulo de la velocidad
get_wind_velmod = lambda x : float(calc.wind_speed(
    x.U * units.meter/units.second, 
    x.V * units.meter/units.second,
).magnitude)

# Obtención de la dirección (meteorológica) del viento
get_wind_dir = lambda x : float(calc.wind_direction(
    x.U * units.meter/units.second, 
    x.V * units.meter/units.second, 
    convention="from"
).magnitude)


class DerivedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_time_feat=True, add_cyclic_feat=True):
        self.add_time_feat = add_time_feat
        self.add_cyclic_feat= add_cyclic_feat

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        # Creamos atributos derivados de las componentes de la velocidad
        x_dataset['w_vel'] = x_dataset.apply(get_wind_velmod, axis=1)
        x_dataset['w_dir'] = x_dataset.apply(get_wind_dir, axis=1)
        
        
        if self.add_time_feat:
            # Creamos atributos derivados de la fecha
            x_dataset['month'] = x_dataset['Time'].dt.month
            x_dataset['hour'] = x_dataset['Time'].dt.hour
            x_dataset['day_of_week'] = x_dataset['Time'].dt.dayofweek
            x_dataset['day_of_month'] = x_dataset['Time'].dt.day
            
        if self.add_cyclic_feat:
            # Hour
            x_dataset['hr_sin'] = np.sin(x_dataset['hour'] * (2.* np.pi / 24))
            x_dataset['hr_cos'] = np.cos(x_dataset['hour'] * (2.* np.pi / 24))

            # Day of the week
            x_dataset['wday_sin'] = np.sin(x_dataset['day_of_week'] * (2.* np.pi / 7))
            x_dataset['wday_cos'] = np.cos(x_dataset['day_of_week'] * (2.* np.pi / 7))

            # Month
            x_dataset['mnth_sin'] = np.sin((x_dataset['month']-1) * (2.* np.pi / 12))
            x_dataset['mnth_cos'] = np.cos((x_dataset['month']-1) * (2.* np.pi / 12))
            
            # Wind direction
            x_dataset['wdir_sin'] = np.sin(x_dataset['w_dir'] * (2.* np.pi / 360))
            x_dataset['wdir_cos'] = np.cos(x_dataset['w_dir'] * (2.* np.pi / 360))
            
             
        return x_dataset


def get_col_prefixes(cols):
    
    prefix_lst = []
    regex = 'NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_(?P<weather_var>\w{1,4})' 
    p = re.compile(regex)
    
    for col in cols:
        m = p.match(col)
        col_prefix = 'NWP' + m.group('NWP') + '_' +  m.group('run') + '_' + m.group('fc_day') + '_'
        prefix_lst.append(col_prefix)
    
    prefix_lst = list(OrderedDict.fromkeys(prefix_lst))
        
    return prefix_lst

def get_df_for_eda(df):
    """
        Convert the dataframe (test/train) to an easily manipulate format,
        without changing the data itself.      
    """
    # Create a temporal dataframe
    df_tmp = pd.DataFrame([])  

    # Regular expresion to capture the values from the column names
    regex = 'NWP(?P<NWP>\d{1})_(?P<run>\d{2}h)_(?P<fc_day>D\W?\d?)_' 
    p = re.compile(regex)
    
    # Get prefix list
    cols = df.columns[3:-1] 
    prefix_lst = get_col_prefixes(cols)
    
    for prefix in prefix_lst:
        
        # Create a second temporal dataframe
        df_tmp2 = pd.DataFrame(np.nan, index=df.index, columns=[
            'WF',
            'NWP',
            'fc_day',
            'run',
            'id_target',
            'time',
            'U',
            'V',
            'T',
            'CLCT',
            'production'   
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