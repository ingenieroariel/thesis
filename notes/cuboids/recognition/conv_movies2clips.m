% Converts between representations of behavior (mat -> avi).
%
% See RECOGNITION_DEMO for general info.
%   [datadir(set_ind)/namei.avi] --> [datadir(set_ind)/clip_namei.mat]
%
% INPUTS
%   set_ind     - set index, value between 0 and nsets-1
%
% See also RECOGNITION_DEMO, CONV_MOVIES2DIVX, CONV_CLIPS2MOVIES

function conv_movies2clips( set_ind )
    srcdir = datadir(set_ind);
    dircontent = dir( [srcdir '\*.avi'] );
    nfiles = length(dircontent);
    if(nfiles==0) warning('No avi files found.'); return; end;
    ticstatusid = ticstatus('converting movies to clips');
    for i=1:nfiles
        fname = dircontent(i).name;
        M = aviread( [srcdir '\' fname] );
        I = movie2images( M );
        clipname = fname(1:end-4);
        cliptype = clipname(1:end-3);
        save( [srcdir '\clip_' clipname '.mat'], 'I', 'clipname', 'cliptype' );
        tocstatus( ticstatusid, i/nfiles );
    end;
