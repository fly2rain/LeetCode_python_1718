function GrO0 = F_simuInitStates (md, NSmp, datIdx)
% Breif:
%    sample the states at time 0 times from the initial distribution of states
%    for 3 simulation data.
%
% Input parameters ################################################
%   Lo:     (1 x 1) the number of elements in one context vector
%   N:      (1 x 1) the number of samples to get.
%   Lov:    (1 x 1) the number of values for the 
%   datIdx: (1 x 1) the choice of simulation model: #1 continuous, #2 discrete
%           and #3 mixed
% 
% Output parameters ################################################
%   GrO0:   (Lo x N) the sampled samples from the context distribution at
%           time point zeros. 
%
% References:
%     [1] Meeting Aug 13th 2015: Batch Off-policy Actor-critic, Peng nFeaiao.
%     [2] Susan's draft on 09/29/2015
%
%  version 1.0 - 12/29/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 3 % the default dataset is the discrete, i.e. #2.
    datIdx = 2;
end

if datIdx == 1 || datIdx == 5 || datIdx==6 % data #1: continuous simulation data =========
    Mu    = zeros (md.Lo, 1);
    Sigma = [   1,      0.3,    -0.3; ...
                0.3,    1,      -0.3;  ...
               -0.3,   -0.3,     1  ];
        
    if md.Lo < 3
        error ('the number of elements in Stat must be greater or equal to 3');
    else
        Sigma = blkdiag ( Sigma, eye (md.Lo-3) );
    end
    
    GrO0 = mvnrnd ( Mu, Sigma, NSmp ).'; % Group of S0
    %     GrO0 = F_removeMean4UnitScale (GrO0);
elseif datIdx == 2 || datIdx==4 % data #2: discrete simulation data =========
    GrO0 = randi ( md.Lov, md.Lo, NSmp ); % start the seeds
elseif datIdx==3 % data #3: the mixed simulation data =========
    GrO0 = rand ( md.Lo, NSmp ); % start the seeds
    %     GrO0 = F_removeMean4UnitScale (GrO0);
else
    error ('Error: datIdx must be in 1, 2 or 3, 4, 5.');
end

% remove the mean and unify the scale for each variable
end