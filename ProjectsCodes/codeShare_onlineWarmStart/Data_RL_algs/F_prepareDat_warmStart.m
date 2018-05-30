function [GrBetaBC, GrBetaOL, GrX, GrY, GrA, GrR]  = F_prepareDat_warmStart (varargin)
% Breif:
%       The Long term average reward to test the actor-critic algorithms
%
% Input parameters ################################################
% GrX    ... (LxTxN) the current states for all the individuals
% GrY    ... (LxTxN) the next states for all the individuals
% GrR:   ... (TxN)   the immediate reward for all the individuals
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

[datIdx,T,alpha,md,nseRwd,nseSt,nseBt,Beta,rngVal,NPeoBC,NPeoOL] = parse_opt (varargin, ...
    'datIdx',6,'TT',42,'alpha',1,'mydim',[],'noiseRwd',1,'noiseStat',0.1, ...
    'noiseBeta',0.005,'Beta',[],'rngVal',10,'NPeoBC',40,'NPeoOL',20);
L = md.Lo;
T_1 = T - 1;
LBeta = size (Beta, 2);
% NPeo = NPeoSub * NGroup;
GrX = zeros (L, T_1, NPeoBC);
GrY = zeros (L, T_1, NPeoBC);
GrA = zeros (T_1, NPeoBC);
GrR = zeros (T_1, NPeoBC);
%
rng (rngVal, 'twister');
GrBetaBC = repmat(Beta, [NPeoBC,1]) + nseBt*randn(NPeoBC, LBeta);
rng (2*rngVal, 'twister');
GrBetaOL = repmat(Beta, [NPeoOL,1]) + nseBt*randn(NPeoOL, LBeta);

% tmp = GrBetaOL - GrBetaBC;
% max ( tmp(:) ) 
% 
% randn (2)
% randn (2)
% if nargout <=2, we only need the beta; otherwise we also need the 
if nargout > 2 
    % generate the initial seeds
    rng (rngVal, 'twister');
    GrO0 = F_simuInitStates (md, NPeoBC, datIdx);
    % generate the trajectories for each user  according to each beta =====
    for nn   = 1 : NPeoBC
        beta = GrBetaBC (nn, :);
        [On, An, Rn] = F_drawDatTupleFromSystem (GrO0(:,nn), 'datIdx',datIdx,'TT',T, ...
            'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,'noiseStat',nseSt,'beta',beta.');
        GrX (:,:,nn) = On (:, 1:T_1);
        GrY (:,:,nn) = On (:, 2:T);
        GrA (:, nn)  = An (1:T_1);
        GrR (:, nn)  = Rn (1:T_1);
    end
end
end
