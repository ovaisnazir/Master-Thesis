
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


def save_fig(path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def export_reports(name, reports, loc):
    """ Export each report in 'reports' to html in the location indicated by 'loc'
    """
    for key in reports.keys():
        try:
            reports[key].to_file(
                output_file = loc + '{}_NWP{}.html'.format(name, key)      
            )
        except Exception:
            print('WARN: Exportation failed for NWP{}'.format(key))
            continue