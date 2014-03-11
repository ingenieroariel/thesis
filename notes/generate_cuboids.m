function save_file = generate_cuboids()
    WORKDIR = pwd;
    addpath(genpath(fullfile(WORKDIR, 'toolbox/')));
    addpath(genpath(fullfile(WORKDIR, 'cuboids/')));

    % The video driver does not like AVI on macs.
    if (ismac)
        extension = 'mov';
    else
        extension = 'AVI';
    end
    
    video_path = fullfile(WORKDIR, sprintf('%s.%s', 'subway_entrance_turnstiles', extension));

    save_file = stfeatures_long_video(video_path, WORKDIR);
    exit()
