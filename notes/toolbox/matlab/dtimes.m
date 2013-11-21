function c = dtimes(a,b)
% dtimes  array multiply similar to .*, but handles some non-equal matrix sizes
%
% c = dtimes(a,b)
% 
% this function also handles cases where the a and b are the same or singleton 
% in all dimentions (regular .* only allows all dims same or all singleton (scalar))
%
% e.g.
%  c = dtimes(ones([3 4 1 5]), ones([1 1 10 5])); would return a 
%  3-by-4-by-10-by-5 matrix of ones.
  
  sa=size(a); 
  sb=size(b);
  sa = [sa ones(1,max(0, length(sb)-length(sa)))];
  sb = [sb ones(1,max(0, length(sa)-length(sb)))];
  if(length(a)==1 || length(b)==1 || all(sa==sb))
    c=a.*b;
    return;
  end

  sr = max(sa,sb);
  assert(all(sa==sb | sa==1 | sb==1), 'dtimes: sizes of a and b dont match.');
  
  c = repmat(a, sr./sa).*repmat(b, sr./sb);
  