function [GrOt, GrAt, GrRt] = F_drawDatTupleFromSystem (O0N, varargin)
% Breif:
%       The Long term average reward to test the actor-critic algorithms
%
% Input parameters ################################################
% O0N    ... (Lo x N) shows N samples from the initial distribution of context.
% Input variables stored in "varargin" -----------------
% datIdx ... (1 x 1) the choice of simulation model: #1 continuous, #2 discrete and #3 mixed
% TT     ... (1 x 1) the ending time points to test the long  term average rewards
% tau:   ... (1 x 1) the parameter for the treatment fatigue effect of Ot,3
% alpha  ... (1 x 1) the parameter to balance the problem more like pure bnadit or Reinforment leanring
% sigma  ... (1 x 1) the noise level when simulating the context and the reward;
% beta   ... (12 x1) 12 parameters to simulate contexts and rewards; the
%           8th ~ 12th elements in beta to construct rewards.
% Output parameters ################################################
% GrOt   ... (Lo x TT x NN) the observed states at every time point for each people
% GrAt   ... (TT x NN) the observed actions at each time point for each people
% GrRt   ... (TT x NN) tbe immediate reward at each time point for each people
%
% References:
%     [1]
%
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

[datIdx,TT,alpha,md,nseRwd,nseSt,beta] = parse_opt (varargin, ...
    'datIdx',2,'TT',100,'alpha',1,...
    'mydim',[],'noiseRwd',1,'noiseStat',0.1,'beta',[]);

% if isempty (thetaN)
thetaN = zeros( md.Lgo, size(O0N,2) );
% end

% data #1: continuous simulation data =========
if datIdx == 1 || datIdx == 3 || datIdx == 5 || datIdx==6
    % set the beta according to the given alpha --------------
    banditBeta    = zeros (3, 1);
    beta([3,5,6]) = (1-alpha)*banditBeta + alpha*beta([3,5,6]);
end

NN    = size (O0N, 2);
% inter-variable to help programming or debuging
GrAt  = zeros (TT, NN); % GrAt to collect everytime's action
GrRt  = zeros (TT, NN); % GrReward to collect everytime's reward
GrOt  = zeros (md.Lo, TT, NN);

%% draw the samaple from the environment ----------------------------------
OtN   = O0N;
AtN   = [];
for t = 1 : TT
    % simulate the context, action and the corresponding reward.
    if datIdx == 1  % data #1: continuous simulation data
        [OtN,AtN,RtN] = F_update1_OtAtRt (OtN,AtN,beta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 2 % data #2: discrete simulation data
        [OtN,AtN,RtN] = F_update2_OtAtRt (OtN,AtN,alpha,beta,...
            nseRwd,thetaN,md,t);
    elseif datIdx == 3 % data #3: the mixed simulation data
        [OtN,AtN,RtN] = F_update3_OtAtRt (OtN,AtN,beta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 4 % data #4: the discrete simulation data that Bandit performs badly
        [OtN,AtN,RtN] = F_update4_OtAtRt (OtN,AtN,alpha,thetaN,md,t);
    elseif datIdx == 5 % data #5: the
        [OtN,AtN,RtN] = F_update5_OtAtRt (OtN,AtN,beta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 6 % data #6: the continous simulation data
        [OtN,AtN,RtN] = F_update6_OtAtRt (OtN,AtN,beta,nseRwd,nseSt,...
            thetaN,md,t);
    end
    
    GrAt(t, :)    = AtN;
    GrRt(t, :)    = RtN;
    GrOt(:, t, :) = OtN;
end
end