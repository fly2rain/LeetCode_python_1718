function [GrOt, GrAt, GrRt] = F_addOutlines2Dat ( GrOt, GrAt, GrRt, ratio, strength)
%
% breif:
%      adding various noise to certain percentage of samples, such as the
%      'gaussian', 'laplace', 'poission', and 'salt & pepper' noises.
%
% Notation:
% GrOt ... (LO x T x N) states at T time points for N people.
% GrAt ... (T x N) the action at T time points for N people. \
% GrAt ... (T x N) the immediate reward for N people.
% ratio ...(1x1) the ratio samples of each class to be noised.
% strength ... (1x1) the multiplier of noise strength.
%
% author: feiyun Zhu, 2014-07-28
%

if nargin < 4
    if nargin < 3
        ratio = 0.1;
    end
    strength = 1;
end

[Lo, T, N] = size (GrOt);
NFea = Lo;
NSmp = T * N;

% the temporary matrices or vectors
X = reshape (GrOt, [NFea, NSmp]);
A = reshape (GrAt, [1, NSmp]);
R = reshape (GrRt, [1, NSmp]);

maxX = mean (X(:));
maxR = mean (R(:));

% choose the samples to add the noises
NchsSmps = ceil (NSmp*ratio);
chsIdxs  = randperm (NSmp,  NchsSmps);

% adding the noise randomly from 4 types of noises.
X(:,chsIdxs) = strength*maxX*laprnd(Lo,NchsSmps) +  X(:,chsIdxs);
R(chsIdxs)   = strength*maxR*laprnd(1,NchsSmps)  +  R(chsIdxs);
A(chsIdxs)   = rand (1, NchsSmps) > 0.5;

%
GrOt = reshape (X, [Lo, T, N]);
GrRt = reshape (R, [T, N]);
GrAt = reshape (A, [T, N]);
end