# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:02:42 2024

"""

import numpy as np
import pandas as pd
from scipy.signal import get_window
import matplotlib.pyplot as plt

# # Example data: counts per day
# dates = pd.date_range(start='2024-09-24', end='2024-11-20')
# counts = [np.random.randint(0, 10) for _ in range(len(dates))]
# data = pd.DataFrame({'date': dates, 'counts': counts})

# # Parameters
# window_size = 6
# hop_size = 3
def moving_average (dates, counts, window_size = 6 , hop_size = 3 ):
    
    data = pd.DataFrame({'date': dates, 'counts': counts})
    # Create a triangular or Hamming window
    # Choose between 'triang' (triangular) or 'hamming'
    window = get_window('triang', window_size)
    
    # Normalize the window so the sum equals 1 (preserve scale)
    window /= window.sum()
    
    # Apply the moving average using the window
    smoothed_counts = np.convolve(data['counts'], window, mode='same')
    
    # Downsample the result using the hop size
    smoothed_counts_downsampled = smoothed_counts[::hop_size]
    downsampled_dates = data['date'][::hop_size]
    
    # Create a DataFrame for the smoothed and downsampled data
    smoothed_data = pd.DataFrame({'date': downsampled_dates, 'smoothed_counts': smoothed_counts_downsampled})
    
    return smoothed_data