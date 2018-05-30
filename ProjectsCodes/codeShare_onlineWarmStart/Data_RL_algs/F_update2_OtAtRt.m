function [OtN,AtN,RtN,OtVN,GotN] = F_update2_OtAtRt (OtN,AtN,alpha,beta,...
    sigma,thetaN,md,t)
% Breif:
%     Update the states, action and reward at t-th time point for the
%     discrete simulation dataset, #2 dataset.
%
% Input parameters ################################################
% OtN 	... (Lo x N) N input state indexs.
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% beta	... (8 x1) 8 elements in beta to simulate rewards. the 8th is the
%           treatment fatigue effect of Ot,3
% sigma ... (1 x 1) the noise level when simulating the reward;
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
%  version 1.0 - 12/24/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

%% 1 Observe context ot by stochastic linear system.
if t ~= 1
    OtN = F_simuStat2 (OtN, AtN, alpha);
end
OtVN = OtN * 0.2;
% context feature for policy function
GotN = F_feaPolicy (OtVN, md.Lgo);

%% 2 Draw action At according to current policy function
AtN  = F_simuAction ( GotN, thetaN );

%% 3 Simulate the immediaite reward rt
RtN = F_simuReward2 (OtVN, AtN, beta, sigma);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the states %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OtN = F_simuStat2 (O0N, AtN, alpha)
% Breif:
%      simulate the next stat given current stat and action for simulation data #2
%       i.e., the discrete simulation dataset 
%
% Input parameters ################################################
% O0N 	... (Lo x N) the input N context vectors with indexs
% AtN   ... (1 x N) the input N actions to update the contexts
% alpha ... (1 x 1) the parameter to balance the problem more like pure bnadit or Reinforment leanring
%
% Output parameters ################################################
% OtN 	... (Lo x N) the updated N context vectors with indexs
%
% References:
%     [1] Constructing Just-in-Time Adaptive Interventions, Peng Liao.
%
%  version 1.0 - 12/08 /2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 3
    alpha = 1;
end

[Lo, N] = size (O0N); % number of feature element in Ot; No Ot
La   = 2;
Lov  = 5;  % number of value for each feature element.

% mat_TransMatrix = ['matStat_TransMatrix alpha=' num2str(alpha)];
% if ~exist ([mat_TransMatrix '.mat'], 'file')
PrGr = zeros (Lov, Lov, La, Lo);
% -- Pr1 -------------------------------
Pr1_01 = [ % the transition matrix of S_{t,2} given A_t=0
    0.52,  0.26,   0.13,   0.06,   0.03; ...
    0.21,  0.42,   0.21,   0.11,   0.05; ...
    0.10,  0.20,   0.40,   0.20,   0.10; ...
    0.05,  0.11,   0.21,   0.42,   0.21; ...
    0.03,  0.06,   0.13,   0.26,   0.52  ...
    ];
% sum (Pr1_01, 2)
nEle = 1;
ia=1;   PrGr(:,:,ia,nEle) = cumsum(Pr1_01, 2);
ia=2;   PrGr(:,:,ia,nEle) = PrGr(:,:,ia-1,nEle);

% -- Pr2 -------------------------------
Pr2_0 = [ % the transition matrix of S_{t,2} given A_t=0
    0.90,  0.10,   0.00,   0.00,   0.00; ...
    0.62,  0.31,   0.07,   0.00,   0.00; ...
    0.38,  0.38,   0.19,   0.05,   0.00; ...
    0.28,  0.28,   0.28,   0.13,   0.03; ...
    0.22,  0.22,   0.22,   0.22,   0.12  ...
    ];
Pr2_1 = [ % the transition matrix of S_{t,2} given A_t=1
    0.1,  0.22,   0.22,   0.22,   0.24; ...
    0.12,  0.13,   0.25,   0.25,   0.25; ...
    0.00,  0.17,   0.17,   0.33,   0.33; ...
    0.00,  0.00,   0.25,   0.25,   0.50; ...
    0.00,  0.00,   0.00,   0.33,   0.67  ...
    ];
Pr2_cha = Pr2_1 - Pr2_0;
Pr2_1   = Pr2_0 + alpha*Pr2_cha;
% sum (Pr2_0, 2)  + sum (Pr2_1, 2)
nEle = 2;
ia=1;   PrGr(:,:,ia,nEle) = cumsum(Pr2_0, 2);
ia=2;   PrGr(:,:,ia,nEle) = cumsum(Pr2_1, 2);

% -- Pr3 -------------------------------
Pr3_0 = [ % the transition matrix of S_{t,3} given A_t=0
    0.80,   0.18,   0.02,   0.00,   0.00; ...
    0.21,   0.42,   0.21,   0.11,   0.05; ...
    0.10,   0.20,   0.40,   0.20,   0.10; ...
    0.05,   0.11,   0.21,   0.42,   0.21; ...
    0.03,   0.06,   0.13,   0.26,   0.52 ...
    ];
Pr3_1 = [ % the transition matrix of S_{t,3} given A_t=1
    0.05,   0.80,   0.15,   0.00,   0.00; ...
    0.00,   0.53,   0.27,   0.13,   0.07; ...
    0.00,   0.00,   0.57,   0.29,   0.14; ...
    0.00,   0.00,   0.00,   0.67,   0.33; ...
    0.00,   0.00,   0.00,   0.09,   0.91  ....
    ];
Pr3_cha = Pr3_1 - Pr3_0;
Pr3_1   = Pr3_0 + alpha*Pr3_cha;
% sum (Pr3_0, 2) + sum (Pr3_1, 2)
nEle = 3;
ia=1;   PrGr(:,:,ia,nEle) = cumsum(Pr3_0, 2);
ia=2;   PrGr(:,:,ia,nEle) = cumsum(Pr3_1, 2);

%% get the new states
OtN = zeros (Lo , N);
AtN = logical(AtN); % find the At == 1
% from the first element to the last element
for k = 1 : Lo
    Okk    = O0N (k, :);      % the first element
    cum_P  = PrGr(Okk,:,1,k); % the cum-probabilty with At=0
    cumTmp = PrGr(Okk,:,2,k); % the cum-probabilty with At=1
    
    cum_P(AtN,:) = cumTmp(AtN,:);
    stN = F_samplingCum (cum_P.');
    OtN(k,:) = stN;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RtNi, RtN] = F_simuReward2 (OtNV, AtN, beta, sigma)
% Breif:
%      simulate the reward given current state and action for data #2, i.e. the
%      discrete simulation data
%
% Input parameters ################################################
% OtNV 	... (L x N) N input state vectors, .
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% beta	... (8 x1) 8 elements in beta to simulate rewards. the 8th is the
%           treatment fatigue effect of Ot,3
% sigma ... (1 x 1) the noise level when simulating the reward;
%
% Output parameters ################################################
% RtN   ... (1 x N) the ouput reward with noises.
%
% References:
%     [1] Constructing Just-in-Time Adaptive Interventions, Peng Liao.
%
%  version 1.0 - 12/09/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 4
    sigma = 0;
end

N = size (OtNV, 2);
if (length(AtN) ~= N)
    error ('action does not have the same length with stats.\n');
end

% simulate the noise in the context and rewards.
% gs = 0.75 * ( OtNV(3,:) - 0.2 );
gs = OtNV(3,:);
RtN = beta(1) .* ( ...
    beta(2) + beta(3)*OtNV(1,:) + beta(4)*OtNV(2,:) + ... % first
    AtN.*( beta(5)*OtNV(1,:) + beta(6)*OtNV(2,:) - beta(7)*(OtNV(3,:)>=0.75) )...
    )  -  beta(8).*gs;

rNoise  = sigma * randn (1, N);
RtNi = RtN + rNoise;
end