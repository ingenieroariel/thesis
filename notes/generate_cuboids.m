
function save_file = generate_cuboids()
    addpath(genpath('/home/x/toolbox/'));
    addpath(genpath('/home/x/cuboids/'));

    video_path='/home/x/Videos/D-ZBcmcje_s.avi';

    save_file = stfeatures_long_video(video_path, '/home/x/Videos/output/');
    exit()
