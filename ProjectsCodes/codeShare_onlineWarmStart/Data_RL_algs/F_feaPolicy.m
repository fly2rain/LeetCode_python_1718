function GotN = F_feaPolicy (OtN, Lgo)
% Breif:
%    Construct the features for the policy function based on the state vectors 
%    simultaneously for all the N samples.
%
% Input parameters ################################################
% OtN  ... (Lo x N) the state feature vectors for N samples
% Lgo  ... (1 x 1) the length of policy feature for each sample .
% 
% Output parameters ################################################
% GotN ... (Lgo x N) the constructed feature for policy function for all the N samples
%
% References:
%     [1]
%
%  version 1.0 - 12/09/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

[Lo, N] = size(OtN); % OtN contains N samples, each with Lo varibles
if nargin < 2
    Lgo=1+Lo;
end

if Lgo == 1+Lo
    GotN = vertcat ( ones(1,N), OtN );
elseif Lgo == 1+Lo+Lo*(Lo-1)/2
    GotN = vertcat ( ...
        ones(1,N), OtN, ...
        OtN(1,:).*OtN(2,:), OtN(1,:).*OtN(3,:), OtN(2,:).*OtN(3,:) ...
        );
else
    error ('Error: datIdx must be 1 or 2.\n');
end
