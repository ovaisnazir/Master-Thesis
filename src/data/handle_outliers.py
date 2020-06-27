# -*- coding: utf-8 -*-

import os
import numpy as np
import click
from metpy import calc
from metpy.units import units
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from make_dataset import import_data 


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
    X['Production'] = y['Production']

    # Calculate wind velocity module
    X['vel'] = X.apply(get_wind_velmod, axis=1)

    # Build data matrix
    X1 = X['vel'].values.reshape(-1,1)
    X2 = X['Production'].values.reshape(-1,1)
    X_ = np.concatenate((X1,X2), axis=1)

    # Using DBSCAN to find outliers
    outlier_detection = DBSCAN(eps = 0.27, metric = "mahalanobis", algorithm = 'brute', min_samples = 20, n_jobs = -1)
    clusters = outlier_detection.fit_predict(X_)
    outliers = np.where(clusters == -1)[0]
    
    # Plot outliers
    plt.scatter(*X_.T, color='yellowgreen', marker='.', label='Inliers')
    plt.scatter(*X_[outliers].T, color='gold', marker='.', label='Outliers')
    plt.legend(loc='lower right')
    plt.xlabel("wind speed [m/s]")
    plt.ylabel("wind power [MWh]")
    save_fig(fig_id, folder, wf)

    return outliers


def export_data(folder, wf, X_train, X_test, y_train, y_test):
    """Export data frames to specified folder."""
    os.makedirs(folder + '/{}/clean'.format(wf), exist_ok=True)
    X_train.to_csv(folder + '{}/clean/X_train.csv'.format(wf), index=False)
    X_test.to_csv(folder + '{}/clean/X_test.csv'.format(wf), index=False)
    y_train.to_csv(folder + '{}/clean/y_train.csv'.format(wf), index=False)
    y_test.to_csv(folder + '{}/clean/y_test.csv'.format(wf), index=False)
    
    
     
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('fig_output_filepath', type=click.Path())
@click.argument('wf')
def main(input_filepath, output_filepath, fig_output_filepath, wf):
    """ 
    Cleans outliers from data/interim data sets using DBSCAN algorithm,
    based on wind power curve (power vs wind velocity).
    """
    # Import data 
    X_train = import_data(input_filepath + wf +  '/X_train.csv')
    y_train = import_data(input_filepath + wf + '/y_train.csv')
    X_test = import_data(input_filepath + wf +  '/X_test.csv')
    y_test = import_data(input_filepath + wf + '/y_test.csv')
    
    # Outliers in train sets
    fix_negative_clct(X_train)
    outliers_train = find_outliers(X_train, y_train, "outliers_train", fig_output_filepath, wf)
    plt.close()
    X_train.drop(X_train.index[list(outliers_train)], inplace=True)
    y_train.drop(y_train.index[list(outliers_train)], inplace=True)
    
    # Outliers in test sets
    fix_negative_clct(X_test)
    outliers_test = find_outliers(X_test, y_test, "outliers_test", fig_output_filepath, wf)
    X_test.drop(X_test.index[list(outliers_test)], inplace=True)
    y_test.drop(y_test.index[list(outliers_test)], inplace=True)
    
    # Save clean data
    export_data(output_filepath, wf, X_train, X_test, y_train, y_test)
    
if __name__ == '__main__':
    main()