% Get location of data; alter file depending on location of dataset.
%
% INPUTS
%   set_ind     - [optional] set index, value between 0 and nsets-1
%
% OUTPUTS
%   dir         - final directory 
%
% EXAMPLE
%   dir = getclipsdir( 'mouse00', 'features' )

function dir = datadir( set_ind )

    % root directory
    %dir = 'C:/code/mice/data';
    dir = 'C:/code/faces/data';
    
    % set index
    if( nargin==0 ) return; end;
    set_ind_str =['set' int2str2(set_ind,2)];
    dir = [dir '/' set_ind_str];
    
