function [GrX, GrY, GrA, GrR, GrC, GrBeta]  = F_prepareDat_adaptiveIntervention (varargin)
% Breif:
%       The Long term average reward to test the actor-critic algorithms
%
% Input parameters ################################################
% GrX    ... (LxTxN) the current states for all the individuals
% GrY    ... (LxTxN) the next states for all the individuals
% GrR:   ... (TxN)   the immediate reward for all the individuals
% GrC    ... (N)     the class label of all the data.
% Output parameters ################################################
% GrOt   ... (Lo x TT x NN) the observed states at every time point for each people
% GrAt   ... (TT x NN) the observed actions at each time point for each people
% GrRt   ... (TT x NN) tbe immediate reward at each time point for each people
%
% References:
%     [1]
%
% version 1.0 -- 05/22/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

[datIdx,T,alpha,md,nseRwd,nseSt,nseBt,Beta,rngVal,NPeoSub] = parse_opt (varargin, ...
    'datIdx',6,'TT',42,'alpha',1,'mydim',[],'noiseRwd',1,'noiseStat',0.1, ...
    'noiseBeta',0.005,'Beta',[],'rngVal',10,'NPeoSub',10);
L = md.Lo;
T_1 = T - 1;
[NGroup, LBeta] = size (Beta);
NPeo = NPeoSub * NGroup; 

GrX = zeros (L, T_1, NPeo);
GrY = zeros (L, T_1, NPeo);
GrA = zeros (T_1, NPeo);
GrR = zeros (T_1, NPeo);
GrC = zeros (NPeo,1);
% parepare beta and class lables ==========================================
GrBeta  = zeros (NPeo, LBeta);
subIdxs = 1:NPeoSub;
for nn  = 1 : NGroup
    % prepare the beta for every individual
    beta = Beta (nn, :);
    idxRange = subIdxs + (nn-1)*NPeoSub; 
    GrBeta (idxRange,:) = repmat(beta,[NPeoSub,1]) + ...
        nseBt*randn(NPeoSub, LBeta);
    % the class label for each individual
    GrC(idxRange) = nn;
end

% generate the initial seeds 
rng (rngVal);
GrO0 = F_simuInitStates (md, NPeo, datIdx);
% generate the trajectories according to each beta ========================
for nn   = 1 : NPeo
    beta = GrBeta (nn, :);
    [On, An, Rn] = F_drawDatTupleFromSystem (GrO0(:,nn), 'datIdx',datIdx,'TT',T, ...
        'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,'noiseStat',nseSt,'beta',beta.');
    GrX (:,:,nn) = On (:, 1:T_1);
    GrY (:,:,nn) = On (:, 2:T);
    GrA (:, nn)  = An (1:T_1);
    GrR (:, nn)  = Rn (1:T_1);
end
end
