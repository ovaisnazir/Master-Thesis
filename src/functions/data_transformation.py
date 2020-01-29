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


def get_age_category(year):
    age = ''
    if year <= 1940:
        age = 'old'
    elif (year > 1940) and (year <= 1980):
        age = 'middle'
    else:
        age = 'new'
        
    return age
    
def add_building_age_feature(df):
    return df.assign(building_age = df['year_built'].apply(get_age_category))