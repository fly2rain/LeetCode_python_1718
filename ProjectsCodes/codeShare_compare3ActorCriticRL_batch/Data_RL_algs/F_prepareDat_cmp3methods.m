function [GrX, GrY, GrA, GrR]  = F_prepareDat_cmp3methods (varargin)
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

[datIdx,T,alpha,md,nseRwd,nseSt,beta,rngVal,NPeo] = parse_opt (varargin, ...
    'datIdx',6,'TT',42,'alpha',1,'mydim',[],'noiseRwd',1,'noiseStat',0.1, ...
    'beta',[],'rngVal',10,'NPeo',10);
T_1 = T - 1;

rng (rngVal);
GrO0 = F_simuInitStates (md, NPeo, datIdx);

[GrOt, GrAt, GrRt] = F_drawDatTupleFromSystem (GrO0, 'datIdx',datIdx,'TT',T, ...
    'alpha',alpha, 'mydim',md,'noiseRwd',nseRwd,'noiseStat',nseSt,'beta',beta);

GrX = GrOt (:, 1:T_1, :);
GrY = GrOt (:, 2:T, :);
GrA = GrAt (1:T_1, :);
GrR = GrRt (1:T_1, :);
end
