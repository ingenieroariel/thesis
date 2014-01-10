function save_file = generate_cuboids()
    ARIEL_LAPTOP='/Users/x/work/thesis/notes/';
    ARIEL_DESKTOP='/home/x';

    WORKDIR = ARIEL_LAPTOP;

    addpath(genpath(strcat(WORKDIR, 'toolbox/')));
    addpath(genpath(strcat(WORKDIR, 'cuboids/')));

    video_path=strcat(WORKDIR, 'D-ZBcmcje_s.mov');

    save_file = stfeatures_long_video(video_path, WORKDIR);
    exit()
