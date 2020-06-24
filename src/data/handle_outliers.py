# -*- coding: utf-8 -*-

import os
import numpy as np
import click
from src.timer import timer
from metpy import calc
from metpy.units import units
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# lambda function to get wind velocity module
get_wind_velmod = lambda x : float(calc.wind_speed(
    x.U * units.meter/units.second, 
    x.V * units.meter/units.second
).magnitude)



def fix_negative_clct(df):
    """Replaces negative values of CLCT by 0."""
    df.loc[df['CLCT'] < 0, 'CLCT'] = 0.0


def find_outliers(X, y):
    """Finds outliers based on power curve, using DBSCAN algorithm."""
    
    # Add 'Production' column
    X['Production'] = y.to_list()

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
    
    return outliers


def plot_outliers(X, outliers):
    """Plots outliers and saves the figure in data/reports/figures."""
    fig = plt.scatter(*X.T, s=50, linewidth=0, c='gray', alpha=0.20)
    fig = plt.scatter(*X[outliers].T, s=50, linewidth=0, c='red', alpha=0.5)
    
    return fig


def save_fig(fig_id, folder, WF, tight_layout=True, fig_extension="png", resolution=300):
    os.makedirsos.makedirs(folder + '/' + WF, exist_ok=True)
    path = os.path.join(folder + '/' + WF, fig_id + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()
        
    plt.savefig(path, format=fig_extension, dpi=resolution)


def delete_outliers(X, y, outliers):
    """Deletes outliers from X (train or test) and y (labels)."""
    if type(y) != 'numpy.ndarray':
        y = y.to_numpy()
    
    if type(X) != 'numpy.ndarray':
        X = X.to_numpy()
    
    X = np.delete(X, tuple(outliers), axis=0)
    y = np.delete(y.to_numpy(), list(outliers))
    
    
     
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('fig_output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Cleans ouliers from data/interim data sets using DBSCAN algorithm,
    based on wind power curve (power vs wind velocity).
    """
    
    
    
    
if __name__ == '__main__':
    main()