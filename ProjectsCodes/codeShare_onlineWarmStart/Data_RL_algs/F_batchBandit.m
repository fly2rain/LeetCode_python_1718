function [theta, w] = F_batchBandit (aaa, rrr, X, Y, varargin)
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
[maxIter,theta,w,md,zetaC,zetaA,OptAlg,valBnd,RBFbnd,dispRst] = parse_opt...
    (varargin,'maxIter',100,'theta',[],'w',[],'mydim',[],'zetaC',0.1,'zetaA',0.1, ...
    'OptAlg','fmincon','valBnd',10,'RBFbnd',1,'dispRst',0);

[Lo, N]   = size (X); % NSmp's data, each is a Lo vector.
if isempty (theta) % initialize the parameter
    theta = zeros (md.Lgo, 1);
end
if isempty (w) % initialize the parameter
    w = zeros (md.Lho, 1);
end
if gamma < 0 || maxIter <= 0 % apply the random policy and random value approx.
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
        error('Invalid optimiz. choice. Choose ''fmincon'', ''fminconGrad'', ''GD'' ');
end % ----------------------------------------------------------------

% value function feature via basic function & policy feature
XX = F_fot ( X, md, RBFbnd );
YY = F_fot ( Y, md, RBFbnd );
    
YG  = F_feaPolicy ( Y, md.Lgo );
XG  = F_feaPolicy ( X, md.Lgo );
XXX = F_feaValueApprox (XX, aaa.', md.Lho);
% iteratively update the critic & actor ===================================
XX0 = F_feaValueApprox (XX, zeros(1,N), md.Lho);
XX1 = F_feaValueApprox (XX, ones(1,N), md.Lho);
for iter = 1 : maxIter
    % %%%% critic updates %%%%
    w = F_criticUpdate ( theta, XXX, YY, YG, rrr, gamma, zetaC );
    % %%%% actor updateS  %%%%
    myfun = @(theta) F_actorUpdate ( theta, w, XG, XX0, XX1, zetaA, md.Lgo );
    if (strcmpi (OptAlg, 'GD')) % the Gradient Descent Method (GD)
        % theta = F_GradDescent (myfun, theta, r, NIter, Tol);
        theta = F_GradDescent (myfun, theta);
    else % the fmincon & fminunc
        [theta,f,exitflag,output] = fmincon (myfun,theta,[],[],[],[],lb,ub,[],opts);
    end
    if dispRst
        fprintf ('[%d]\tf:%f\n', iter, f);
    end
end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = F_criticUpdate ( theta, XXX, YY, YG, rrr, gamma, zetaC )
% Breif:
%       critic update for DisRwd via LSTDQ
%
% Input parameters ################################################
%
% Output parameters ################################################
%
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% get the next state's feature
[Lho, NT]  = size (XXX);
expThetaGo = exp (theta' * YG);
Pi  = expThetaGo ./ ( 1 + expThetaGo );
YYY = F_feaValueApprox ( YY, Pi, Lho );

if gamma < 1 && gamma >=0 % the contextual bandit & the discount reward method
    % critic update for vt
    w = ( zetaC*eye(Lho) + 1/NT*(XXX*(XXX-gamma*YYY).') ) \ (1/NT*(XXX*rrr) );
elseif gamma == 1  % the average reward method ============================
    XXAvg = XXX - repmat (mean(XXX,2), [1, NT]);
    % critic update for vt
    w = ( zetaC*eye(Lho) + 1/NT*(XXAvg*(XXX-gamma*YYY).') ) \ (1/NT*(XXAvg*rrr) );
end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, g] = F_actorUpdate (theta, w, XG, XX0, XX1, zetaA, Lgo)
% Breif:
%       actor update for DisRwd via fmincon
%
% Input parameters ################################################
%
% Output parameters ################################################
%
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% #1 Pi ----------------------------
expThetaGo = exp (theta.' * XG); % 1 x t
Pi_0  = 1 ./ (1 + expThetaGo);   % 1 x t
Pi_1  = 1 - Pi_0;                % 1 x t

Q_A0 = w' * XX0;    % 1 x t
Q_A1 = w' * XX1;    % 1 x t

f  = - mean ( (Q_A0.*Pi_0) + (Q_A1.*Pi_1), 2 ); % min 2 max
f  = f + (zetaA/2)*(theta' * theta);

if nargout > 1 % the gradient g is required
    % #2 d\pi/d\theta ---------------
    dPi_0 = - (expThetaGo .* Pi_0.^2);  % 1 x t
    dPi_1 = - dPi_0;
    
    g = - mean ( XG .* repmat( (Q_A0.*dPi_0)+(Q_A1.*dPi_1), Lgo, 1), 2 );
    g = g + zetaA*theta;
end
end