# Pain_GGN_model
High accuracy classification and intensity assessment of pain based on EEG data

# 1. Raw Data Slicing
The MATLAB script Pain_EEG_data_slice.m is used to slice the raw EEG sequences into 20-second segments, allowing for the filtering and extraction of specific electrodes. This step also includes downsampling the data to 200Hz and applying average referencing.

# 2. Data Preprocessing
Using the mff2npy_preprocess_all_in_one.ipynb notebook, the raw EEG signals are divided into time-windowed spectral arrays via a sliding window FFT method. The resulting data dimensions are (sample, time_windows, channels, frequency). The adjustable parameters in this process include window length, window overlap degree, and the maximum/minimum frequency range.

# 3. Model Training and Testing
Model training is performed in gcn_main.py, which supports both classification and regression tasks. For the classification task, the feature arrays and label arrays for all samples are loaded for training and testing. For the regression task, only the feature arrays of pain samples are loaded, along with their corresponding NRS and VAS pain scores, for training and testing.

# 4. Data Availability
The complete features and labels for the Pain GGN model (including pain categories and NRS/VAS pain intensity scores) can be accessed at the following link: https://doi.org/10.6084/m9.figshare.30744833. Three raw EEG files in .mff format are attached as example data; you may apply to us for other required data in accordance with the regulations.
