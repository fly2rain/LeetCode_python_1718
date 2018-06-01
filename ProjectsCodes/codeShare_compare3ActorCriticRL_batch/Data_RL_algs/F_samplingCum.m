function atN = F_samplingCum (cumP)
% Breif:
%     Sample N times from the N discrete distribution in cumP
%
% Input parameters ################################################
%   cumP:  (K x N) the cumulative distribution, each column is a statistic
%          variable's probability.
% 
% Output parameters ################################################
%   atN:   (1 x N) the sampled indexes from the inut distribution.
%
% version 2.0 -- 12/09/2015
% version 1.0 -- 10/09/2015
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

[K, N] = size (cumP);
if K < 2
    error ('Must: 2 or more probablity elements. \n');
end

atN = ones (1, N);
randVal = rand(1, N);

for ii = 1 : K - 1
    cumPii  = cumP (ii, :);
    cmpRst  = randVal > cumPii;
    atN (cmpRst) = ii + 1;
end
end