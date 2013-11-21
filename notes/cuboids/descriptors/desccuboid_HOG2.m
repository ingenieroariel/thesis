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


function desc = desccuboid_HOG2(I, ch2params)

if ndims(I)~=4
    error('I must be MxNxTx2');
end;

if ~isa(I,'double') 
    error('I must be of type double'); 
end;

%%% gradient images must be given in input
Weights = repmat(I(:,:,:,1), [1 1 1 1 1]);
GS = repmat(I(:,:,:,2), [1 1 1 1 1]);

%%% call imagedesc_ch2desc
desc = imagedesc_ch2desc({GS, Weights}, ch2params, 1, 1, 1);
