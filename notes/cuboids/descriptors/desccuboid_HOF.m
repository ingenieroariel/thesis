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

function desc = desccuboid_HOF( I, flow_params, angle_edges, ch2params )
if ndims(I)~=3
    error('I must be MxNxT');
end;
if ~isa(I,'double')
    error('I must be of type double');
end;

%%% get optical flow in channels
siz = size(I);  nframes = siz(3);
Fx=zeros([siz(1:2) nframes-1]);  Fy=Fx;  reliab=Fx;
for i=1:nframes-1
    [Fx(:,:,i), Fy(:,:,i)] = ...
        optFlowLk(I(:,:,i), I(:,:,i+1), flow_params{:});
end;

F_small = (Fx.*Fx + Fy.*Fy) < 0.4*0.4;

Fx = repmat(Fx, [1 1 1 size(angle_edges, 2)]);
Fy = repmat(Fy, [1 1 1 size(angle_edges, 2)]);

angle_x = reshape(angle_edges(1,:), 1, 1, 1, []);
angle_y = reshape(angle_edges(2,:), 1, 1, 1, []);

% dot product
angle_dot_product = dtimes(Fx, angle_x) + dtimes(Fy, angle_y);
[~, F_angle] = max(angle_dot_product, [] ,4);
F_angle(F_small) = 0;

%%% call imagedesc_ch2desc  [will always have 2 channel, 1 instance!]
desc = imagedesc_ch2desc( F_angle, ch2params, 1, 1, 1 );
