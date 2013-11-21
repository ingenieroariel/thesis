% Cuboid descriptor based on histogrammed gradient.
%
% Adaptation of Lowe's SIFT descriptor for cuboids.  Creates a descriptor for an cuboid
% that is fairly robust to small perturbations of the cuboid.  No histogramming (if
% histflag==-1) See "PCA-SIFT: A More Distinctive Representation for Local Image
% Descriptors" by Yan Ke for why this might be a good idea. Should not be called directly,
% instead use imagedesc.
%
% INPUTS
%   I               - MxNxT double array (cuboid) with most vals in range [-1,1]
%   sigmas          - n-element vector of spatial scales at which to look at gradient
%   taus            - n-element vector of temporal scales at which to look at gradient
%   ch2params       - see imagedesc_ch2desc
%   ignGt           - if 1 the temporal gradient is ignored
%
% OUTPUTS
%   desc            - 1xp feature vector, where p=n*prod(size(cuboid))
%
% See also IMAGEDESC, IMAGEDESC_CH2DESC


function desc = desccuboid_HOG(I, sigmas, angle_edges, ch2params)

if ndims(I)<3
    error('I must be MxNxT');
end;

if ndims(I)==4 && size(I,3)~=3
    error('I must be MxNx3xT');
end;

if ndims(I)>4
    error('I must be MxNxT');
end

if ~isa(I,'double') 
    error('I must be of type double'); 
end;
colorImage = false;
if ndims(I)==4 && size(I,3)==3
    colorImage = true;
end

%%% create gradient images
nsigmas = length(sigmas);
for s = 1:nsigmas
    if colorImage
        I_r = permute(I(:,:,1,:), [1 2 4 3]);
        L_r = gaussSmooth(I_r, [sigmas(s) sigmas(s) 0.1], 'same', 2);
        [Gx_r, Gy_r] = gradient(L_r);
        G_mag_r = sqrt(Gx_r.*Gx_r + Gy_r.*Gy_r);
        
        I_g = permute(I(:,:,2,:), [1 2 4 3]);
        L_g = gaussSmooth(I_g, [sigmas(s) sigmas(s) 0.1], 'same', 2);
        [Gx_g, Gy_g] = gradient(L_g);
        G_mag_g = sqrt(Gx_g.*Gx_g + Gy_g.*Gy_g);
        
        I_b = permute(I(:,:,3,:), [1 2 4 3]);
        L_b = gaussSmooth(I_b, [sigmas(s) sigmas(s) 0.1], 'same', 2);
        [Gx_b, Gy_b] = gradient(L_b);
        G_mag_b = sqrt(Gx_b.*Gx_b + Gy_b.*Gy_b);
        G_mag = max(cat(5, G_mag_r, G_mag_g, G_mag_b), [], 5);
        error('color videos not supported yet');
    else
        L = gaussSmooth(I, [sigmas(s) sigmas(s) 0.1], 'same', 2);
        [Gx, Gy] = gradient(L);
        G_mag = sqrt(Gx.*Gx + Gy.*Gy);
    end
    Gx = repmat(Gx, [1 1 1 size(angle_edges, 2)]);
    Gy = repmat(Gy, [1 1 1 size(angle_edges, 2)]);
    
    angle_x = reshape(angle_edges(1,:), 1, 1, 1, []);
    angle_y = reshape(angle_edges(2,:), 1, 1, 1, []);
    
    % dot product
    angle_dot_product = abs(dtimes(Gx, angle_x) + dtimes(Gy, angle_y));
    [dummy, G_angle] = max(angle_dot_product, [] ,4);
        
    if s == 1
        GS = repmat(G_angle,[1 1 1 1 nsigmas]);
        Weights = repmat(G_mag,[1 1 1 1 nsigmas]);
    else
        GS(:,:,:,:,s) = G_angle;
        Weights (:,:,:,:,s) = G_mag;
    end;
end;

%%% call imagedesc_ch2desc
desc = imagedesc_ch2desc({GS, Weights}, ch2params, 1, 1, nsigmas);
