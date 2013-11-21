% Cuboid descriptor based on histogrammed optical flow.
%
% INPUTS
%   I               - MxNxT double array (cuboid) with most vals in range [-1,1]
%   flow_params     - paramters for lucaskanade optical flow (see optflow_lucaskanade)
%   ch2params       - see imagedesc_ch2desc
%
% OUTPUTS
%   desc            - 1xp feature vector
%
% See also IMAGEDESC, OPTFLOW_LUCASKANADE, IMAGEDESC_CH2DESC

function desc = desccuboid_HOF2( I, ch2params )
if ndims(I)~=3
    error('I must be MxNxT');
end;
if ~isa(I,'double')
    error('I must be of type double');
end;

%%% call imagedesc_ch2desc  [will always have 2 channel, 1 instance!]
desc = imagedesc_ch2desc(I, ch2params, 1, 1, 1 );
