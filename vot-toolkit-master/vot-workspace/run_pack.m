% This script can be used to pack the results and submit them to a challenge.

addpath('C:\Users\xyd\Desktop\VOT2015-KCF\vot-toolkit-master'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

tracker = tracker_load('KCF');

workspace_submit(tracker, sequences, experiments);

