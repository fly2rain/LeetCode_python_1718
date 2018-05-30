function [OtN,AtN,RtN,GotN] = F_update3_OtAtRt (OtN,AtN,beta,...
    nseRwd, nseSt,thetaN,md,t)
% Breif:
%     Update the states, action and reward at t-th time point for the
%     mixed simulation dataset, #3 dataset.
%
% Input parameters ################################################
% OtN 	... (Lo x N) N input state vectors.
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% beta	... (13 x1) 13 parameters to simulate contexts and rewards for the
%           simulation data #1, the continuous data; the 8th ~ 12th elements 
%           in beta to simulate rewards. 13th is the treatment fatigue effect of Ot,3
% sigma ... (1 x 1) the noise level when simulating the reward;
% thetaN... (Lgo x n) the parameter in the policy function for all the n samples
% Lgo   ... (1 x 1) the length of policy feature for each sample.
% t     ... (1 x 1) the time point
%
% Output parameters ################################################
% OtN 	... (Lo x N) the updated N context vectors 
% AtN 	... (1 x N) the updated N actions.
% RtN   ... (1 x N) the updated N reward with noises.
% GotN  ... (Lgo x N) the updated feature for the policy function.
%
%  version 1.0 - 12/09/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

%% 1 Observe context ot by stochastic linear system.
if t ~= 1
    OtN = F_simuStat3 ( OtN, AtN, beta, nseSt );
end

% context feature for policy function
GotN = F_feaPolicy ( OtN, md.Lgo );

%% 2 Draw action At according to current policy function
AtN = F_simuAction ( GotN, thetaN );

%% 3 Simulate the immediaite   reward rt: both rewards are ok
RtN = F_simuReward3 (OtN, AtN, beta, nseRwd); % test bandit first
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the states %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OtN = F_simuStat3 (O0N, AtN, beta, nseSt)
% Breif:
%      simulate the next stat given current stat and action for data #1
%       i.e., the continuous simulation dataset.
%
% Input parameters ################################################
% O0N 	... (Lo x N) the input N context vectors 
% AtN   ... (1 x N) the input N actions to update the contexts
% % beta... (12 x1) 12 parameters to simulate contexts for the simulation
%           data #1, the continuous data; 
% sigma ... (1 x 1) control the strength of noise
%
% Output parameters ################################################
% OtN 	... (Lo x N) the updated N context vectors 
%
% References:
%     [1] Meeting Aug 13th 2015: Batch Off-policy Actor-critic, Peng nFeaiao.
%     [2] Susan's draft on 09/29/2015
%
%  version 1.0 - 10/10/2015
%  version 1.0 - 11/01 /2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)
% to balance the magnitude between states and rewards.
if nargin < 4
    nseSt = 0;
end
if nargin < 3 % default parameter in beta
    beta = [0.3,0.2,0.8,0.3,0.05,0.9,0.25,1000,1,0.2,0.6,0.3,0.7,0.4].';
end

[Lo, NN] = size (O0N); % length of feature.
if (NN ~= length(AtN) )
    error('action a0 does not have the same length with stats s0.\n');
end
OtN      = zeros (Lo, NN);
% update the st given O0N and AtN
rNoi     = nseSt * randn (Lo, NN); % randn(nFea, nSmp);
OtN(1,:) = beta(1)*O0N(1,:) + rNoi(1,:);
OtN(2,:) = beta(2)*O0N(2,:) + beta(3)*AtN + rNoi(2,:);
OtN(3,:) = beta(4)*O0N(3,:) + beta(5)*(AtN.*O0N(3,:)) ...
    + beta(6)*AtN + rNoi(3,:);

if Lo > 3
    OtN(4:end,:) = beta(7)*O0N(4:end,:) + rNoi(4:end,:);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [RtNi, RtN] = F_simuReward3 (OtN, AtN, beta, sigma)
% Breif:
%      simulate the reward given current state and action for data #2, i.e. the
%      discrete simulation data
%
% Input parameters ################################################
% OtNV 	... (L x N) N input state vectors, .
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% beta	... (13 x1) 13 parameters to simulate contexts and rewards for the
%           simulation data #1, the continuous data; the 8th ~ 12th elements
%           in beta to simulate rewards. 13th is the treatment fatigue effect of Ot,3
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

N = size (OtN, 2);
if (length(AtN) ~= N)
    error ('action does not have the same length with stats.\n');
end

% simulate the noise in the context and rewards.
RtN = beta(8) + 0.5*beta(8).*( ...
    AtN.*( beta(9) + beta(10)*OtN(1,:) + beta(11)*OtN(2,:) ) +  ...
    beta(12)*OtN(1,:) - beta(13).*OtN(3,:)   ...
);
% RtN = beta(8) + AtN.*( beta(9) + beta(10)*OtN(1,:) + beta(11)*OtN(2,:) ) ...
%       + beta(12)*OtN(1,:) - beta(13)*OtN(3,:); % for evalueation

rNoise  = sigma * randn (1, N);
RtNi    = RtN + rNoise;
end