function save_file = generate_cuboids()
    %ARIEL_LAPTOP='/Users/x/work/thesis/notes/';
    %ARIEL_DESKTOP='/home/x';

    WORKDIR = pwd;
    addpath(genpath(fullfile(WORKDIR, 'toolbox/')));
    addpath(genpath(fullfile(WORKDIR, 'cuboids/')));

    video_path = fullfile(WORKDIR, 'D-ZBcmcje_s.avi');

    save_file = stfeatures_long_video(video_path, WORKDIR);
    exit()
