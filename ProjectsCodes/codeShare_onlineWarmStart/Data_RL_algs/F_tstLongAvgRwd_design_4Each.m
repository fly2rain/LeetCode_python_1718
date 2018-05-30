function longRwd = F_tstLongAvgRwd_design_4Each (O0N, rndPy, varargin)
% Breif:
%       The Long term average reward to test the actor-critic algorithms
%
% Input parameters ################################################
% O0N    ... (Lo x N) shows N samples from the initial distribution of context.
% thetaN ... (Lgo x N) the parameters for the policy function  for N people.
% Input variables stored in "varargin" -----------------
% datIdx ... (1 x 1) the choice of simulation model: #1 continuous, #2 discrete and #3 mixed
% TT,T0  ... (1 x 1) the starting & ending time points to test the long  term average rewards
% tau:   ... (1 x 1) the parameter for the treatment fatigue effect of Ot,3
% alpha  ... (1 x 1) the parameter to balance the problem more like pure bnadit or Reinforment leanring
% sigma  ... (1 x 1) the noise level when simulating the context and the reward;
% Lgo    ... (1 x 1) the number of elements in the g(o), which is a feature for policy function.
% beta   ... (12 x1) 12 parameters to simulate contexts and rewards; the
%           8th ~ 12th elements in beta to construct rewards.
% Output parameters ################################################
% longRwd... (1 x 1) the long term average reward in the time range of T0 to TT.
%
% References:
%     [1]
%
% version 1.0 -- 05/24/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)
[datIdx,TT,T0,alpha,nseRwd,nseSt,GrBeta] = parse_opt (varargin, ...
    'datIdx',2,'EndTime',10^4,'StarTime',10^3,'alpha',1,...
    'nseRwd',1,'nseSt',0.1,'GrBeta',[]);

NPeo = size (O0N, 2);
GrBeta = GrBeta.';
% continuous simulation data =========
if datIdx == 1 || datIdx == 3 || datIdx == 5 || datIdx==6
    % set the beta according to the given alpha --------------
    banditBeta = zeros (3, NPeo);
    GrBeta([3,5,6], :) = (1-alpha)*banditBeta + alpha*GrBeta([3,5,6], :);
end

%% inter-variable to help programming or debuging
GrAt  = zeros (TT,  NPeo); % GrAt to collect everytime's action
GrRt  = zeros (TT,  NPeo); % GrReward to collect everytime's reward

OtN   = O0N;
AtN   = [];
% OtMin = OtN; OtMax = OtN;
for t = 1 : TT
    % simulate the context, action and the corresponding reward.
    if datIdx == 6 % data #6: the continous simulation data
        [OtN,AtN,RtN] = F_update_design_OtAtRt (OtN,AtN,GrBeta,nseRwd,nseSt,...
            rndPy,t);
    end
    %% 2 draw action At according to current policy function
    GrAt(t, :) = AtN;
    GrRt(t, :) = RtN;
end

%% #1 the long term average
GrIdxs = (T0+1) : TT; % selected points
tmpGrRwd = GrRt (GrIdxs, :); % consider the 10^3 to 10^4 reward
longRwd  = mean( tmpGrRwd, 1);
end

function [OtN,AtN,RtN] = F_update_design_OtAtRt (OtN,AtN,beta,...
    nseRwd,nseSt,rndPoly,t)
% Breif:
%     Update the states, action and reward at time point t for data #1
%     i.e., the continuous simulation dataset
%
% Input parameters ################################################
% OtN 	... (Lo x N) N input state vectors.
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% beta	... (13 x1) 13 parameters to simulate contexts and rewards for the
%           simulation data #1, the continuous data; the 8th ~ 12th elements 
%           in beta to simulate rewards. 13th is the treatment fatigue effect of Ot,3
% sigma ... (1 x 1) the noise level when simulating the reward;
% thetaN... (Lgo x N) the parameter in the policy function for all the n samples
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
    OtN = F_simuStat6 (OtN, AtN, beta, nseSt);
end

NPeo = size (OtN, 2);

%% 2 Draw action At according to current policy function
AtN = ones(1,NPeo) * rndPoly;

%% 3 Simulate the immediaite reward rt: both rewards are ok
RtN = F_simuReward6 ( OtN, AtN, beta, nseRwd );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the states %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OtN = F_simuStat6 (O0N, AtN, beta, nseSt)
% Breif:
%     Simulate the next stat given current stat and action
%       i.e., the continuous simulation dataset.
% References:
%     [1] Meeting Aug 13th 2015: Batch Off-policy Actor-critic, Peng nFeaiao.
%     [2] Susan's draft on 09/29/2015
%
%  version 1.0 - 10/10/2015
%  version 1.0 - 11/01 /2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)
if nargin < 4
    nseSt = 0;
end

[Lo, NN] = size (O0N); % length of feature.
if ( NN ~= length(AtN) )
    error('action a0 does not have the same length with stats s0.\n');
end
OtN      = zeros (Lo, NN);
% update the st given O0N and AtN
rNoi     = nseSt * randn (Lo, NN); % randn(nFea, nSmp);
OtN(1,:) = beta(14) + beta(1)*O0N(1,:) + rNoi(1,:);
OtN(2,:) = beta(15) + beta(2)*O0N(2,:) + beta(3)*AtN + rNoi(2,:);
OtN(3,:) = beta(16) + beta(4)*O0N(3,:) + beta(5)*(AtN.*O0N(3,:)) ...
    + beta(6)*AtN + rNoi(3,:);

if Lo > 3
    OtN(4:end,:) = beta(17) + beta(7)*O0N(4:end,:) + rNoi(4:end,:);
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% % simulate the reward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RtN = F_simuReward6 (OtN, AtN, beta, nseRwd)
% Breif:
%     Simulate the immediate reward given current state and action
% 
%  version 1.0 - 10/10/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 5
    nseRwd = 0;
end

nSmp = size (OtN, 2);
if (length(AtN) ~= nSmp)
    error ('action does not have the same length with stats.\n');
end

RtN = beta(8) + AtN.*( beta(9) + beta(10)*OtN(1,:) + beta(11)*OtN(2,:) ) ...
      + beta(12)*OtN(1,:) - beta(13)*OtN(3,:); % for evalueation

% simulate the noise in the context and rewards.
rNoise = nseRwd * randn (1, nSmp);
RtN = beta(18) * ( RtN + rNoise );
end