function [OtN,AtN,RtN,OtVN,GotN] = F_update4_OtAtRt (O0N,AtN,alpha,thetaN,md,t)
% Breif:
%     Update the states, action and reward at t-th time point for the
%     discrete simulation dataset, #2 dataset.
%
% Input parameters ################################################
% OtN 	... (Lo x N) N input state indexs.
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% alpha ... (1 x 1) the parameter that changes the simulation model from
%           the pure bandit problem to the reinforcement learning problem.
% thetaN... (Lgo x n) the parameter in the policy function for all the n samples
% Lgo   ... (1 x 1) the length of policy feature for each sample.
% OtVals... (5 x 1) the values for the state indexs
% t     ... (1 x 1) the time point
%
% Output parameters ################################################
% OtN 	... (Lo x N) the updated N context vectors with indexs 
% AtN 	... (1 x N) the updated N actions.
% RtN   ... (1 x N) the updated N reward with noises.
% OtVN  ... (Lo x N) the updated N context vectors with values
% GotN  ... (Lgo x N) the updated feature for the policy function.
%
% version 1.0 - 12/24/2015
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

%% 1 Observe context ot by stochastic linear system.
if t ~= 1
    OtN = F_simuStat4 (O0N, AtN, alpha);
else
    OtN = O0N;
end
OtVN = OtN * 0.2;
% context feature for policy function
GotN = F_feaPolicy (OtVN, md.Lgo);

%% 2 Draw action At according to current policy function
AtN = F_simuAction ( GotN, thetaN );

%% 3 Simulate the immediaite reward rt
RtN = F_simuReward4 (O0N, OtN);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the states %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OtN = F_simuStat4 (O0N, AtN, alpha, Lov)
% Breif:
%      simulate the next stat given current stat and action for simulation
%      data #4, i.e., the discrete simulation data that Bandit performs badly
%
% Input parameters ################################################
% O0N 	... (Lo x N) the input N context vectors with indexs, here Lo=1
% AtN   ... (1 x N) the input N actions to update the contexts
% alpha ... (1 x 1) the parameter to balance the problem more like pure bnadit or Reinforment leanring
% Lov   ... (1 x 1) the number of states
% Output parameters ################################################
% OtN 	... (Lo x N) the updated N context vectors with indexs
%
% References:
%     [1] Talking with Ambuj on 02/02/2016
%
%  version 1.0 - 02/07/2016
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 4 % number of values in each state element
    Lov = 5; 
    if nargin < 3
        alpha = 1;
    end
end
Lo = size (O0N, 1); % number of feature element in Ot
if Lo > 1
    error ('Lo must be equal to 1.');
end

% set the transition probability -----------------
La = 2; % two actions 
PrGr = zeros (Lov, Lov, La);
Pr0 = [ % the transition matrix of S_t given A_t=0
    0.90,  0.10,   0.00,   0.00,   0.00; ...
    0.80,  0.00,   0.20,   0.00,   0.00; ...
    0.85,  0.00,   0.00,   0.15,   0.00; ...
    0.90,  0.00,   0.00,   0.00,   0.10; ...
    0.95,  0.00,   0.00,   0.00,   0.05  ...
    ];
Pr1 = [ % the transition matrix of S_t given A_t=1
    0.10,  0.90,   0.00,   0.00,   0.00; ...
    0.20,  0.00,   0.80,   0.00,   0.00; ...
    0.15,  0.00,   0.00,   0.85,   0.00; ...
    0.10,  0.00,   0.00,   0.00,   0.90; ...
    0.05,  0.00,   0.00,   0.00,   0.95  ...
    ];
%
Pr_cha = Pr1 - Pr0;
Pr1    = Pr0 + alpha*Pr_cha;
%
ia=1;   PrGr(:,:,ia) = cumsum(Pr0, 2);
ia=2;   PrGr(:,:,ia) = cumsum(Pr1, 2);

%% Get the new states
cum_P  = PrGr(O0N,:,1); % the cum-prob0abilty with At=0
cumTmp = PrGr(O0N,:,2); % the cum-pro1babilty with At=1

AtN = logical (AtN);
cum_P(AtN,:) = cumTmp(AtN,:);
OtN = F_samplingCum (cum_P.');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Rt = F_simuReward4 (O0N, OtN, Lov)
% Breif:
%      simulate the reward given current state and action for data #4, i.e.
%      the discrete simulation data that Bandit performs badly
%
% Input parameters ################################################
% O0N 	... (L x N) the state vector for N individuals at time point t.
% OtN   ... (L x N) the state vector for N individuals at time point t+1.
% Lov   ... (1 x 1) parameter for the treatment fatigue effect of Ot,3
%
% Output parameters ################################################
% Rt    ... (1 x N) the ouput reward with noises.
%
% References:
%     [1] Talking with Ambuj on 02/02/2016
%
%  version 1.0 - 02/07/2016
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 3
    Lov = 5;
end

% set the reward matrix
% GrR = [ % the transition matrix of S_t given A_t=0
%     0.01,  0.00,   0.00,   0.00,   0.00; ...
%     0.02,  0.00,   0.00,   0.00,   0.00; ...
%     0.02,  0.00,   0.00,   0.00,   0.00; ...
%     0.02,  0.00,   0.00,   0.00,   0.00; ...
%     1.20,  0.00,   0.00,   0.00,   0.90  ...
%     ];

% set the reward matrix
GrR = [ % the transition matrix of S_t given A_t=0
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    1.10,  0.00,   0.00,   0.00,   0.90  ...
    ];

% GrR = [ % the transition matrix of S_t given A_t=0
%     0.01,  0.00,   0.00,   0.00,   0.00; ...
%     1.00,  0.00,   0.00,   0.00,   0.00; ...
%     1.00,  0.00,   0.00,   0.00,   0.00; ...
%     1.00,  0.00,   0.00,   0.00,   0.00; ...
%     1.00,  0.00,   0.00,   0.00,   0.90  ...
%     ];

idxs = O0N + (OtN-1)*Lov;
Rt   = GrR(idxs);
end