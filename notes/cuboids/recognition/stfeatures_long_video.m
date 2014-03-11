function save_file = stfeatures_long_video(video_filename, output_path)

%read in clip
fprintf('Loading in clip\n');
readerobj = VideoReader(video_filename);
clip_length = get(readerobj, 'numberOfFrames');

clip_step = 900;
clip_step_length = 1000;

k = 1 + clip_step_length - clip_step;
k_ind = 0;

while (k < clip_length)
    
    %update k
    k = k + clip_step;
    k_ind = k_ind + 1;
    
    %check to make sure we are not over
    k = min(k, clip_length);
    
    %get start and end frames
    frame1 = double(max(k - clip_step_length, 1)); %in case entire clip is less than 3000 frames
    frame2 = double(k);
    
    %sanity check
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
    save_file = fullfile(output_path, ['features_' filename '_' sprintf('%02d', k_ind) '.mat']);
    save(save_file, 'subs', 'vals', 'frame1', 'frame2','desc_hog', 'desc_hof');
    
end
