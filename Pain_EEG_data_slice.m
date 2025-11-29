clear all; clc;

group_dir = 'P:\Pain_EEG\X1';   
save_dir = 'P:\Pain_EEG\sliced\X1';

group_files = dir([group_dir, filesep, '*.mff']);


% Electrode selection
% Collect the electrodes corresponding to the 10-20 lead system in the 128 EGI electrode system
% FP1 FP2 F3 F4 C3  C4 P3 P4 O1 O2  F7 F8 T3 T4 T5  T6 A1 A2 Fz Cz  Pz  # T1 T2
select_channels = [22 9 24 124 36 104 52 92 70 83 33 122 45 108 58 96 56 107 11 129 62 39 115];

originalChannels = select_channels(1:21);
[~, originalIndices] = sort(originalChannels);
[~, reverseIndices] = sort(originalIndices);

segmentLength = 20 * 200;
% Sampling rate of 200Hz, with each segment length of 20s

eeglab;
EEG.etc.eeglabvers = '2023.1';

for i= 1 : length(group_files)

    subj_fn = group_files(i).name;
    
    % EEG = loadcurry(strcat(group_dir, filesep, subj_fn), 'CurryLocations', 'False'); 
    EEG = pop_mffimport(strcat(group_dir, filesep, subj_fn),{},0,0);
    % The imported raw data is multi-channel time series data in. mff format,
    % Each sample ranging from 200 to 300 seconds
    
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.1);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',75);
    EEG = pop_eegfiltnew(EEG, 'locutoff',49,'hicutoff',51,'revfilt',1);
    
    EEG = pop_reref(EEG, [], 'exclude', []);
    % Average reference for all channels

    EEG = pop_resample(EEG, 200);
    
    EEG = pop_select(EEG, 'channel', select_channels);   

    start_time = 5; 
    end_time = (EEG.pnts / 200) - 15;
    EEG = pop_select(EEG, 'time', [start_time end_time]);
    % Extract the beginning and end of signals with significant interference

    data = EEG.data(1:21,:); 
    data = data(reverseIndices, :);

    [numChannels, numSamples] = size(data);

    numSegments = floor(numSamples / segmentLength);

    for k = 1:numSegments
        startIndex = (k - 1) * segmentLength + 1;
        endIndex = k * segmentLength;
        segment = data(:, startIndex:endIndex);     
        filename = fullfile(save_dir, [group_files(i).name(1:end-4), sprintf('_%d.mat', k)]);
        save(filename, 'segment');
    end
    % Slice the raw data into 20 second time periods without overlap and save them as. mat files separately

    EEG = pop_saveset( EEG, 'filename',strcat(group_files(i).name(1:end-4),'_final', '.set'), 'filepath',strcat(group_dir, filesep, subj_fn));

    ALLEEG = []; 
    EEG = []; 
    CURRENTSET = []; 

end

disp("done!")
