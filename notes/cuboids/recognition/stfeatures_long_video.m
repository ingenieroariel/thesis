function save_file = stfeatures_long_video(video_filename, output_path)

% scale_ind varies from 1 to 16
%i varies from 1 to 26

%read in clip
fprintf('Loading in clip\n');
readerobj = VideoReader(video_filename);
clip_length = get(readerobj, 'numberOfFrames');

k = clip_length;

%get start and end frames
frame1 = 1; %in case entire clip is less than 3000 frames
frame2 = double(k);
        
%sanity check
disp([frame1 frame2 clip_length]);
        
mov = read(readerobj, [frame1 frame2]);
        
fprintf('Converting to grayscale\n');
mov_bw = zeros(size(mov,1), size(mov,2), size(mov,4), 'uint8');
for j = 1:size(mov, 4)
    mov_bw(:,:,j) = rgb2gray(mov(:,:,:,j));
end
        
fprintf('Detecting features\n');
[R,subs,vals,cuboids, V] = stfeatures( mov_bw, 3, 3, 1, 5e-4, [],1.85, 2, 1, 0);
clear R;
% update frames
if ~isempty(subs)
    subs(:,3) = subs(:,3) + frame1 - 1;
end
        
fprintf('Computing descriptors\n');
iscuboid = 1;  histFLAG = 1;  jitterFLAG = 0;
hog_desc = imagedesc_generate( iscuboid, 'HOG', histFLAG, jitterFLAG );
hof_desc = imagedesc_generate( iscuboid, 'HOF', histFLAG, jitterFLAG );
hog = imagedesc(cuboids, hog_desc, 0);
hof = imagedesc(cuboids, hof_desc, 0);
        
        
[dummy, filename] = fileparts(video_filename);
save_file = fullfile(output_path, ['features_' filename '.mat']);
save(save_file, 'subs', 'vals', 'hof', 'hog', 'cuboids', 'V', 'mov_bw', 'mov');
