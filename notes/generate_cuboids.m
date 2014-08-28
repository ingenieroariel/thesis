function generate_cuboids(video_filename, output_path, start_frame, end_frame, overlap)

% Initialize the libraries this script depends on.
WORKDIR = pwd;
addpath(genpath(fullfile(WORKDIR, 'toolbox/')));
addpath(genpath(fullfile(WORKDIR, 'cuboids/')));

%read in clip
fprintf('Loading in clip\n');
disp(video_filename);
readerobj = VideoReader(video_filename);
clip_length = get(readerobj, 'numberOfFrames');

frame1 = double(max(1, start_frame - overlap));
frame2 = double(min(clip_length, end_frame + overlap));

%sanity check
fprintf('Processing only frames within the following boundaries\n');
disp([frame1 frame2 clip_length]);

mov = read(readerobj, [frame1 frame2]);

fprintf('Converting to grayscale\n');
mov_bw = zeros(size(mov,1), size(mov,2), size(mov,4), 'uint8');
for j = 1:size(mov, 4)
    mov_bw(:,:,j) = rgb2gray(mov(:,:,:,j));
end
clear mov;

fprintf('Detecting features\n');
[R,subs,vals,cuboids] = stfeatures( mov_bw, 1.5, 3, 1, 1e-3, [],1.85, 2, 1, 0);
clear R;
% update frames
if ~isempty(subs)
    subs(:,3) = subs(:,3) + frame1 - 1;
end

fprintf('Computing descriptors\n');
iscuboid = 1;  histFLAG = 1;  jitterFLAG = 0;
imdesc_hog = imagedesc_generate( iscuboid, 'HOG', histFLAG, jitterFLAG );
desc_hog = imagedesc(cuboids, imdesc_hog, 0);

imdesc_hof = imagedesc_generate( iscuboid, 'HOF', histFLAG, jitterFLAG );
desc_hof = imagedesc(cuboids, imdesc_hof, 0);

[dummy, filename] = fileparts(video_filename);
save_file = fullfile(output_path);
save(save_file, 'subs', 'vals',  'frame1', 'frame2','desc_hog', 'desc_hof');

exit()
