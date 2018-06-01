function theta = F_batchRL_substitute (aaa, rrr, X, Y, varargin)
% Breif:
%    Actor-critic discount reward method & average reward method
%      #1 critic update: LSTDQ;       #2 actor update: Fmincon, as Susan's draft
%
% Input parameters ################################################
% aaa    ... (NSmp) the action of N people at T time points
% rrr    ... (NSmp) the immediate reward of N people at T time points
% X      ... (Lo x NSmp) the set of current states
% Y      ... (Lo x NSmp) the set of next states
% Input variables stored in "varargin" -----------------
% T,T0:     (1 x 1) T0 means the accumulated time points, T means the total training time points
% theta:    (Lgo x 1) the input parameters for the policy function for all the np-th people, which is defaultly set as zero.
% zeta:     (1 x 1) is the strength of \ell_{2}-constraint on v, which is a parameter for the value function
% Lgo:      (1 x 1) the number of elements in g(o), which is a feature for policy function.
% Lho:      (1 x 1) the number of elements in h(o), which is the feature for value function.
% Lfo:      (1 x 1) the number of elements in f(o), which is a basic function for the construction of the feature for value funciton.
% OptAlg:   to choose the optm. algorithm, whish should be one in {fmincon,fminconGrad,fminunc,fminuncGrad,GD}
% gamma:    (1 x 1) a known discount factor, which is set as 0 for the contextual bandit method
% dispRst:  (1 x 1) to choose display the debugging results or not. If true, print the results, and vice versa.
% valbnd:   (1 x 1) the box constraint on theta, which is set in the range from  -valbnd to valbnd
%
% Output parameters ################################################
% theta: (Lgo x 1) the learnt parameter for the policy function.
% vt:    (Lho x 1) the learnt parameter for the value funciton.
%
% References:
%     [1] Lecture 7: Policy Gradient. David Silver, 2015.
%     [2] ActCritic_fyzhu_v6 Section 8.3
%
% version 1.0 -- 03/12/2015
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)
% varargin = varargin{1};
[gamma,maxIter,theta,md,zetaC,zetaA,OptAlg,valBnd,RBFbnd,dispRst] = parse_opt...
    (varargin,'gamma',0,'maxIter',100,'theta',[],'mydim',[],'zetaC',0.1,'zetaA',0.1, ...
    'OptAlg','fmincon','valBnd',10,'RBFbnd',1,'dispRst',0);

[Lo, N]   = size (X); % NSmp's data, each is a Lo vector.
if isempty (theta)
    theta = zeros (md.Lgo, 1);
end

% if gamma < 0, random policy and random value approx.
if gamma < 0
    theta = zeros (md.Lgo, 1);
    % w     = zeros (md.Lho, 1);
    return;
end
% Options for the optimization algorithms from Matlab -----------------
lb  = -valBnd * ones (md.Lgo, 1);
ub  =  valBnd * ones (md.Lgo, 1);
switch OptAlg % ------------------------------------------------------
    case 'fmincon'     % fmincon: interior-point or sqp
        opts = optimoptions(@fmincon,'Algorithm','sqp', 'GradObj','off'); % box constrained quasi-newton
    case 'fminconGrad' % fmincon + gradient
        opts = optimoptions(@fmincon,'Algorithm','sqp', 'GradObj','on'); % box constrained quasi-newton
    case 'GD'          % gradient decent
        opts = [];
    otherwise
        error('Invalid optimiz. choice for Bandit. Choose ''fmincon'', ''fminconGrad'', ''fminunc'', ''fminuncGrad'' ');
end % ----------------------------------------------------------------

% value function feature via basic function & policy feature
if md.Lfo == md.Lo
    XX = X;
    YY = Y;
else
    XX = F_fot ( X, md, RBFbnd );
    YY = F_fot ( Y, md, RBFbnd );
end
YG  = F_feaPolicy ( Y, md.Lgo );
% XG  = F_feaPolicy ( X, md.Lgo );
XXX = F_feaValueApprox (XX, aaa.', md.Lho);
YY0 = F_feaValueApprox (YY, zeros(1,N), md.Lho);
YY1 = F_feaValueApprox (YY, ones(1,N), md.Lho);

% iteratively update the critic & actor ===================================
if gamma < 1 % the contextual bandit & the discount reward method
    for iter = 1 : maxIter
        % %%%% critic & actor updateS  %%%%
        myfun = @(theta) F_criticActorUpdate_dis ( theta, YG, YY0, YY1, XXX, ...
            YY, rrr, gamma, zetaC, zetaA );
        [theta,f] = fmincon (myfun,theta,[],[],[],[],lb,ub,[],opts);
        
        if dispRst
            fprintf ('[%d]\tf:%f\n', iter, f);
        end
    end
elseif gamma == 1  % the average reward method ============================
    for iter = 1 : maxIter
        % %%%% critic & actor updates %%%%
        myfun = @(theta) F_criticActorUpdate_avg ( theta, YG, YY0, YY1, XXX, ...
            YY, rrr, gamma, zetaC, zetaA );
        [theta,f] = fmincon (myfun,theta,[],[],[],[],lb,ub,[],opts);
        
        if dispRst
            fprintf ('[%d]\tf:%f\n', iter, f);
        end
    end
end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, w] = F_criticActorUpdate_dis ( theta, YG, YY0, YY1, XXX, YY, rrr, gamma, zetaC, zetaA )
% Breif:
%       critic & actor update for DisRwd via LSTDQ
%
% Input parameters ################################################
%
% Output parameters ################################################
%
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% get the next state's feature
Lho = size (XXX, 1);
expThetaGo = exp (theta' * YG);
Pi_0  = 1 ./ (1 + expThetaGo);   % 1 x t
Pi_1  = 1 - Pi_0;
YYY = F_feaValueApprox ( YY, Pi_1, Lho );
% critic update for vt
w = ( zetaC*eye(Lho) + XXX*(XXX-gamma*YYY).' ) \ (XXX*rrr);

% #1 Pi ----------------------------
Q_A0 = w' * YY0;    % 1 x t
Q_A1 = w' * YY1;    % 1 x t

f  = - mean ( (Q_A0.*Pi_0) + (Q_A1.*Pi_1), 2 ); % min 2 max
f  = f + (zetaA/2)*(theta' * theta);
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = F_criticActorUpdate_avg ( theta, YG, XXX, YY, rrr, gamma, zetaC, zetaA )
% Breif:
%       critic update for AvgRwd via LSTDQ
%
% Input parameters ################################################
%
% Output parameters ################################################
%
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% get the next state's feature
[Lho, T]   = size (XXX);
% #1 Pi ----------------------------
expThetaGo = exp (theta.' * YG); % 1 x t
Pi_0  = 1 ./ (1 + expThetaGo);   % 1 x t
Pi_1  = 1 - Pi_0;                % 1 x t
YYY   = F_feaValueApprox ( YY, Pi_1, Lho );
XXAvg = XXX - repmat (mean(XXX,2), [1, T]);
% critic update for vt
w = ( zetaC*eye(Lho) + XXAvg*(XXX-gamma*YYY).' ) \ (XXAvg*rrr);

% Q_A0 = w' * YY0;    % 1 x t
% Q_A1 = w' * YY1;    % 1 x t
% Vw   = w' * XXX;
% f  = mean (rrr.' - Vw);
% f  = f + mean ( (Q_A0.*Pi_0) + (Q_A1.*Pi_1), 2 ); % min 2 max
% f  = f + (zetaA/2)*(theta' * theta);
f  = -mean(rrr + (YYY-XXX).' * w, 1) + (zetaA/2)*(theta' * theta);
end