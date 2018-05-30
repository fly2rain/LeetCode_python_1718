    function hotN = F_feaValueApprox (fotN, atN, Lho)
% Breif:
%   Construt the final features for the critic (value) function based on
%   the input basic feature vectors simultaneously for all the N samples.
%  
% Input parameters ################################################
%   fotN: (Lfo x N) the input basic feature for all the N samples
%   atN:  (1 x N) the input actionsfor all the N samples
%   Lho:  (1 x 1) the length of feature vectros for the value function.
% 
% Output parameters ################################################
%   hotN: (Lho x N) the final feature vectors for the value function. 
%
% References:
%     [1]
%
%  version 1.0 - 12/09/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

[Lfo, N] = size(fotN);

if Lho == 2*Lfo + 2     % (1, fot, at, at*fot)
    hotN = vertcat ( ones(1,N), fotN, atN, repmat(atN,Lfo,1).*fotN );
elseif Lho == 2*Lfo + 1 % (fot, at, at*fot)
    hotN = vertcat ( fotN, atN, repmat(atN,Lfo,1).*fotN );
elseif Lho == 2*Lfo
    hotN = vertcat ( (1-repmat(atN,Lfo,1)).*fotN, repmat(atN,Lfo,1).*fotN );
else
    error ('No hot is selected.\n');
end
    