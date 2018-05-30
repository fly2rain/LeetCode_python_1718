function [longRwd, GrQtErr, avgLongRwd, stdLongRwd] = F_tstLongAvgRwd_val_4Each ...
    (O0N, thetaN, vtN, varargin)
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
[datIdx,TT,T0,alpha,md,nseRwd,nseSt,GrBeta,gamma,RBFbnd] = parse_opt (varargin, ...
    'datIdx',2,'EndTime',10^4,'StarTime',10^3,'alpha',1,...
    'mydim',[],'nseRwd',1,'nseSt',0.1,'GrBeta',[],'gamma',0,'RBFbnd',1);

NPeo = size (O0N, 2);
GrBeta = GrBeta.';
% continuous simulation data =========
if datIdx == 1 || datIdx == 3 || datIdx == 5 || datIdx==6
    % set the beta according to the given alpha --------------
    banditBeta = zeros (3, NPeo);
    GrBeta([3,5,6], :) = (1-alpha)*banditBeta + alpha*GrBeta([3,5,6], :);
end

% every individual has his own beta
if (size(GrBeta,2) ~= NPeo) || (NPeo ~= size (thetaN,2))
    error ('Error: every individual should have his own beta.\n');
end


%% inter-variable to help programming or debuging
GrAt  = zeros (TT,  NPeo); % GrAt to collect everytime's action
GrRt  = zeros (TT,  NPeo); % GrReward to collect everytime's reward
GrQtMC = zeros (T0, NPeo);
GrQtHO = zeros (T0, NPeo);
GrHot = zeros (md.Lho, NPeo, T0);

OtN   = O0N;
AtN   = [];
% OtMin = OtN; OtMax = OtN;
for t = 1 : TT
    % simulate the context, action and the corresponding reward.
    if datIdx == 1  % data #1: continuous simulation data
        [OtN,AtN,RtN] = F_update1_OtAtRt (OtN,AtN,GrBeta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 2 % data #2: discrete simulation data
        [OtN,AtN,RtN] = F_update2_OtAtRt (OtN,AtN,alpha,GrBeta,...
            nseRwd,thetaN,md,t);
    elseif datIdx == 3 % data #3: the mixed simulation data
        [OtN,AtN,RtN] = F_update3_OtAtRt (OtN,AtN,GrBeta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 4 % data #4: the discrete simulation data that Bandit performs badly
        [OtN,AtN,RtN] = F_update4_OtAtRt (OtN,AtN,alpha,thetaN,md,t);
    elseif datIdx == 5 % data #5: the
        [OtN,AtN,RtN] = F_update5_OtAtRt (OtN,AtN,GrBeta,nseRwd,nseSt,...
            thetaN,md,t);
    elseif datIdx == 6 % data #6: the continous simulation data
        [OtN,AtN,RtN] = F_update6_OtAtRt (OtN,AtN,GrBeta,nseRwd,nseSt,...
            thetaN,md,t);
    end
    
    %% 2 draw action At according to current policy function
    GrAt(t, :) = AtN;
    GrRt(t, :) = RtN;
    
    if t < T0 + 1
        % transfer the state vector from indexes into values
        if datIdx == 2 || datIdx == 4
            OtNV = 0.2 * OtN;
        else
            OtNV = OtN;
        end
        
        FotN = F_fot ( OtNV,  md,  RBFbnd );
        HotN = F_hot ( FotN,  AtN,  md.Lho );
        GrHot(:,:,t) = HotN;
    end
end

%% #1 the long term average
GrIdxs = (T0+1) : TT; % selected points
tmpGrRwd = GrRt (GrIdxs, :); % consider the 10^3 to 10^4 reward
longRwd  = mean( tmpGrRwd, 1);
if nargout > 2
    avgLongRwd = mean (longRwd);
    stdLongRwd = std (longRwd);
end

%% #2 verify the approximate value funciton via the Monte Carlo of return.
if nargout > 1
    GrEta = repmat (longRwd, [TT, 1]);
    GrWeight = repmat ( (0:TT-1).', [1, NN]);
    for t = 1 : T0
        % the esitmated value via Monte Carlo
        if gamma == 1     % avg Rwd
            tmpGt       = GrRt(t:TT,:) - GrEta(t:TT,:);
            GrQtMC(t,:) = sum (tmpGt, 1);
        elseif gamma  > 0 % discount reward
            tmpWeight   = gamma .^ GrWeight(1:TT-t+1,:);
            tmpGt       = GrRt(t:TT,:) .* tmpWeight;
            GrQtMC(t,:) = sum (tmpGt, 1);
        elseif gamma == 0 % contextual bandit
            GrQtMC(t,:) = GrRt(t, :) ;
        elseif gamma  < 0 % random policy
            GrQtMC(t,:) = 0;
        end
        
        % the esitmated value via value function approximation
        HotN = GrHot (:,:,t);
        GrQtHO (t, :) = sum (HotN .* vtN, 1);
    end
    GrQtErr = mean( (GrQtMC - GrQtHO).^2, 1);
    GrQtErr = GrQtErr .^ 0.5;
end
% boxplot (tmpGrRwd);
% figure,
% boxplot (longRwd)
% % if nargout > 1
% %     GrWeight = 1 ./ (GrIdxs - T0);
% %     GrLongRwd = cumsum (tmpGrRwd, 1);
% %     GrLongRwd = GrLongRwd .* repmat(GrWeight.', 1, NN);
% % end
end