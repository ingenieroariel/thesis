% Generate parameters for image or cuboid descriptor.
%
% INPUTS
%   iscuboid    - 1 for cuboids, 0 for patches
%   str_desc    - 'GRAD', 'FB' for images [see descpatch_*.m]
%               - 'APR', 'GRAD', 'IMDESC', 'WW', 'FLOW' [see desccuboid_*.m]
%               - 'HOG', 'HOF' [see desccuboid_*.m]
%   histFLAG    - see imagedesc_ch2desc, also has some additional values [2,3]
%   jitterFLAG  - [optional] see jitter_dist, causes everything to be very slow
%
% See also IMAGEDESC, IMAGEDESC_DEMO, IMAGEDESC_CH2DESC

function imdesc = imagedesc_generate(iscuboid, str_desc, histFLAG, jitterFLAG)
if( nargin<4 ) jitterFLAG = 0; end;

if( ~iscuboid )  %%% FOR IMAGES:
    imdesc.iscuboid = 0;
    
    % create default ch2params, edges must still be initialized
    if( ~isempty(histFLAG) )
        ch2params.histFLAG=histFLAG;
        if( histFLAG == 0 )
            nbins=12;
            ch2params.pargmask = {[2 2],.65,.1,0};
        else
            nbins=10;
            ch2params.pargmask = {[2 2],.65,.1,0};
        end
    end;
    
    switch str_desc
        case 'GRAD';
            ch2params.edges = quickedges(nbins,.16,1);
            sigmas=[1 2];
            imdesc.par_desc={sigmas,ch2params};
            imdesc.fun_desc = @descpatch_GRAD;
            
        case 'FB';
            ch2params.edges = quickedges(nbins,.2,1);
            load FB;
            imdesc.par_desc={FB,ch2params};
            imdesc.fun_desc = @descpatch_FB;
            
        otherwise error('unknown flag');
    end
    
    if( jitterFLAG )
        imdesc.par_jitter = {1,0,3,3};
    end;
    
else %%% FOR CUBOIDS:
    imdesc.iscuboid = 1;
    
    % create default ch2params, edges must still be initialized
    if( ~isempty(histFLAG) )
        ch2params.histFLAG=histFLAG; nbins=12;
        switch histFLAG
            case -1 % string out
            case 0 %1D - local position-dependent histograms
                nbins = 4;
                ch2params.pargmask = {[3 3 2],.65,.1,0};
            case 1 %3D - local position-dependent histograms
                nbins = 10;
                ch2params.pargmask = {[3 3 2],.65,.1,0};
            case 2 %1D - global histogram
                ch2params.histFLAG=0;
                nbins = 12;
                ch2params.pargmask = {[1],.65,.1,0};
            case 3 %3D - global histogram
                ch2params.histFLAG=1;
                nbins = 12;
                ch2params.pargmask = {[1],.65,.1,0};
            otherwise
                error('not recog histFLAG');
        end
    end;
    
    % create according to switch
    switch str_desc
        case 'HOG'
            ch2params.edges = 1:(nbins+1);
            angle_edges = angleedges(nbins, false);
            sigmas=3;
            imdesc.par_desc = {sigmas, angle_edges, ch2params};
            imdesc.fun_desc = @desccuboid_HOG;
            imdesc.normalizeIm = false;
            
        case 'HOG2'
            ch2params.edges = 1:(nbins+1);
            imdesc.par_desc = {ch2params};
            imdesc.fun_desc = @desccuboid_HOG2;
            imdesc.normalizeIm = false;
            imdesc.hasChn = true;
            imdesc.hog.angle_edges = angleedges(nbins, false);
            imdesc.hog.sigma = 1;
            
        case 'HOF'
            ch2params.edges = 0:(nbins+1);
            angle_edges = angleedges(nbins, true, true);
            flow_params = {[],4,2,[]};
            imdesc.par_desc = {flow_params, angle_edges, ch2params};
            imdesc.fun_desc = @desccuboid_HOF;
            imdesc.normalizeIm = false;
            
        case 'HOF2'
            ch2params.edges = 0:(nbins+1);
            imdesc.hof.angle_edges = angleedges(nbins, true, true);
            imdesc.hof.flow_params = {[],4,2,[]};
            imdesc.par_desc = {ch2params};
            imdesc.fun_desc = @desccuboid_HOF2;
            imdesc.normalizeIm = false;
            
        case 'GRAD'
            ch2params.edges = quickedges(nbins,.16,1);
            sigmas=[1 2];  taus = [.5 .5];  ignGt=0;
            imdesc.par_desc = {sigmas,taus,ch2params,ignGt};
            imdesc.fun_desc = @desccuboid_GRAD;
            imdesc.normalizeIm = true;
            
        case 'APR'
            ch2params.edges = quickedges(12,1.2,1);
            sigmas=[1]; taus=[.5];
            imdesc.par_desc = {sigmas,taus,ch2params};
            imdesc.fun_desc = @desccuboid_APR;
            imdesc.normalizeIm = true;
            
        case 'IMDESC'
            imdesc.fun_desc = @desccuboid_IMDESC;
            imdesc.normalizeIm = true;
            warning('still need to initialze image descriptor');
            
        case 'FLOW'
            ch2params.edges = quickedges(nbins,8,1);
            flow_params = {[],4,2,[]};
            imdesc.par_desc = {flow_params,ch2params};
            imdesc.fun_desc = @desccuboid_FLOW;
            imdesc.normalizeIm = true;
            
        case 'WW'
            warning( 'setting 3rd parameter weight to .2' )
            who_imdesc = imagedesc_generate( 0, 'GRAD', 0 );
            what_imdesc = imagedesc_generate( 1, 'FLOW', 0 );
            weight = .2;
            imdesc.par_desc = { who_imdesc, what_imdesc, weight };
            imdesc.fun_desc = @desccuboid_WW;
            imdesc.normalizeIm = true;
            
        otherwise error('unknown flag');
    end;
    
    if( jitterFLAG )
        imdesc.par_jitter = {1,0,3,3,3,3};
    end;
end
end


function edges = quickedges( nbins, pmax, flag )
switch flag
    case 1
        pmax = pmax / 1.5;
        edges=linspace(-pmax,pmax,nbins+1);
        edges(1)=-inf; edges(end)=inf;
    otherwise
        error('unknown flag');
end
end

function edges = angleedges(nbins, dir_flag, offset_flag)
if nargin<2 ||  isempty(dir_flag)
    dir_flag = false;
end
if nargin<3  ||  isempty(offset_flag)
    offset_flag = false;
end
% if dir_flag is true, will take into account angle sign
if dir_flag == true
    k = 2;
else
    k = 1;
end
% if offset_flag, add offset to angle edges
if offset_flag == true
    angle_offset = pi / nbins;
else
    angle_offset = 0;
end
edges = [cos(angle_offset + k*pi*(0:nbins-1)/nbins)
         sin(angle_offset + k*pi*(0:nbins-1)/nbins)];
     
end
