function [theta, w] = F_onlineRL_warmSt_cum (GrO0, np, datIdx, a0a, r0r, X0, Y0, varargin)
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
[gamma,TT,theta,md,zetaC,zetaA,OptAlg,valBnd,RBFbnd,beta,nseRwd,nseSt,w,dispRst] = parse_opt...
    (varargin,'gamma',0,'TT',100,'theta',[],'mydim',[],'zetaC',0.1,'zetaA',0.1, ...
    'OptAlg','fmincon','valBnd',10,'RBFbnd',1,'beta',[],'nseRwd',1,'nseSt',1,'dispRst',0, ...
    'w',[]);
Tgap = 2;
% [Lo, N]   = size (X); % NSmp's data, each is a Lo vector.
if isempty (theta) % initialize the parameter
    theta = zeros (md.Lgo, 1);
end
if isempty (w) % initialize the parameter
    w = zeros (md.Lho, 1);
end
if gamma < 0 || TT <= 0 % apply the random policy and random value approx.
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
    otherwise
        error('Invalid optimiz. choice. Choose ''fmincon'', ''fminconGrad'' ');
end % ----------------------------------------------------------------

if datIdx ~= 6
    error ('datIdx should be equal to 6.\n');
end

GrAt  = zeros (TT, 1); % GrAt to collect everytime's action
GrRt  = zeros (TT, 1); % GrReward to collect everytime's reward
GrOt  = zeros (md.Lo, TT);
OtN   = GrO0(:,np); AtN = [];
for t = 1 : TT
    %% 1 %%%% Generate trjectory tuples
    [OtN,AtN,RtN] = F_update6_OtAtRt (OtN,AtN,beta,nseRwd,nseSt,...
        theta,md,t);
    GrAt(t)    = AtN;
    GrRt(t)    = RtN;
    GrOt(:, t) = OtN;
    
    if ( ~rem(t, Tgap) && t > 1 )
        %     if ( t > 1 )
        % prepare the datasets
        t_1 = t - 1;
        % prepare the datasets
        aaa = GrAt (1:t_1);
        rrr = GrRt (1:t_1);
        X   = GrOt (:, 1:t_1);
        Y   = GrOt (:, 2:t);
        
        %% 2 the actor-critic RL updates
        [theta, w] = F_actCriticRL (X,Y,aaa,rrr,gamma,md,RBFbnd,theta,t_1,zetaA, ...
            zetaC,lb,ub,opts,OptAlg,dispRst,a0a,r0r,X0,Y0);
    end
end
end

function [theta, w] = F_actCriticRL (X,Y,aaa,rrr,gamma,md,RBFbnd,theta,t_1,zetaA, ...
    zetaC,lb,ub,opts,OptAlg,dispRst,a0a,r0r,X0,Y0)
% Breif:
%      the batch RL learning @ 06/27/2016 (fyzhu0915@gmail.com)

NT = size (X0, 2);
%% 1 for new users' data --------------------------------------
% middel value feature via basic function
XX = F_fot ( X, md, RBFbnd );  % for new user
YY = F_fot ( Y, md, RBFbnd );  % for new user
% final value feature, with history actions
XXA = F_feaValueApprox (XX, aaa.', md.Lho); % for new user
% final value feature, all 0 or 1 actions
XX0 = F_feaValueApprox (XX, zeros(1,t_1), md.Lho);
XX1 = F_feaValueApprox (XX, ones(1,t_1), md.Lho);
% policy feature
YG  = F_feaPolicy ( Y, md.Lgo ); % for new user
XG  = F_feaPolicy ( X, md.Lgo ); % for new user

%% 2 for samples in the 1st dataset ---------------------------
% middel value feature via basic function
X0X = F_fot ( X0, md, RBFbnd ); % for sample in 1st set
Y0Y = F_fot ( Y0, md, RBFbnd ); % for sample in 1st set
% final value feature, with history actions
X0XA = F_feaValueApprox (X0X, a0a.', md.Lho); % for sample in 1st set
% final value feature, all 0 or 1 actions
X0X0 = F_feaValueApprox (X0X, zeros(1,NT), md.Lho);
X0X1 = F_feaValueApprox (X0X, ones(1,NT), md.Lho);
% policy feature
Y0G = F_feaPolicy ( Y0, md.Lgo ); % for sample in 1st set
X0G = F_feaPolicy ( X0, md.Lgo ); % for sample in 1st set

%% 3 %%%% Critic updates %%%%
w = F_criticUpdate ( theta, XXA, YY, YG, rrr, X0XA, Y0Y, Y0G, r0r, gamma, zetaC );
%% 4 %%%% Actor updateS  %%%%
myfun = @(theta) F_actorUpdate ( theta, w, XG, XX0, XX1, X0G, X0X0, X0X1, zetaA, md.Lgo );
if (strcmpi (OptAlg, 'GD')) % the Gradient Descent Method (GD)
    % theta = F_GradDescent (myfun, theta, r, NIter, Tol);
    theta = F_GradDescent (myfun, theta);
else % the fmincon & fminunc
    [theta,f,exitflag,output] = fmincon (myfun,theta,[],[],[],[],lb,ub,[],opts);
end
% if dispRst
%     fprintf ('[%d]\tf:%f\n', t, f);
% end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wt = F_criticUpdate ( theta, XXA, YY, YG, rrr, X0XA, Y0Y, Y0G, r0r, gamma, zetaC )
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
[Lho, T]  = size (XXA);
NT0       = size (X0XA, 2);
% update YYY for current user
expThetaGo = exp (theta' * YG);
Pi  = expThetaGo ./ ( 1 + expThetaGo );
YYA = F_feaValueApprox ( YY, Pi, Lho );

% update Y0YY for the 1st set
expThetaGo0 = exp (theta' * Y0G);
Pi0 = expThetaGo0 ./ ( 1 + expThetaGo0 );
Y0YA = F_feaValueApprox ( Y0Y, Pi0, Lho );

if gamma < 1 && gamma >=0 % the contextual bandit & the discount reward method
    % critic update for wt
    wDenom = zetaC*eye(Lho) + 1/(T+NT0) * ( XXA*(XXA-gamma*YYA).' + ...
        X0XA*(X0XA-gamma*Y0YA).' );
    wNumer = 1/(T+NT0) * ( XXA*rrr + (X0XA*r0r) );
elseif gamma == 1  % the average reward method ============================
    XXAvg  = XXA - repmat (mean(XXA,2), [1, T]);
    X0XAvg = X0XA - repmat (mean(X0XA,2), [1, NT0]);
    % critic update for wt
    wDenom = zetaC*eye(Lho) + 1/(T+NT0) * ( XXAvg*(XXA-gamma*YYA).' + ...
        ( X0XAvg*(X0XA-gamma*Y0YA).') );
    wNumer = 1/(T+NT0) * ( XXAvg*rrr + X0XAvg*r0r );
end
wt = wDenom \ wNumer;
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, g] = F_actorUpdate ( theta, w, XG, XX0, XX1, X0G, X0X0, X0X1, zetaA, Lgo )
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

T   = size (XG, 2);
NT0 = size (XX0, 2);
% for the current user ----------------------------
expThetaGo = exp (theta.' * XG); % 1 x t
Pi_0  = 1 ./ (1 + expThetaGo);   % 1 x t
Pi_1  = 1 - Pi_0;                % 1 x t
% the estimated value
Q_A0 = w' * XX0;    % 1 x t
Q_A1 = w' * XX1;    % 1 x t

% for the 1st set -----------------------------
expThetaGo0 = exp (theta.' * X0G); % 1 x t
Pi0_0  = 1 ./ (1 + expThetaGo0);   % 1 x t
Pi0_1  = 1 - Pi0_0;                % 1 x t
% the estimated value
Q0_A0 = w' * X0X0;    % 1 x t
Q0_A1 = w' * X0X1;    % 1 x t

f  = -1/(T+NT0) * ( ...
    sum( (Q_A0.*Pi_0) + (Q_A1.*Pi_1), 2 ) + ... % min 2 max
    sum( (Q0_A0.*Pi0_0) + (Q0_A1.*Pi0_1), 2 ) ...
    );
f  = f + (zetaA/2)*(theta' * theta);

if nargout > 1 % the gradient g is required
    % #2 d\pi/d\theta ---------------
    % for the current user ----------------------------
    dPi_0 = - (expThetaGo .* Pi_0.^2);  % 1 x t
    dPi_1 = - dPi_0;
    % for the 1st set -----------------------------
    dPi0_0 = - (expThetaGo0 .* Pi0_0.^2);  % 1 x t
    dPi0_1 = - dPi0_0;
    
    g = - 1/(T+NT0) * ( ....
        sum( XG.*repmat( (Q_A0.*dPi_0)+(Q_A1.*dPi_1), Lgo, 1), 2 ) + ...
        sum( X0G .* repmat( (Q0_A0.*dPi0_0)+(Q0_A1.*dPi0_1), Lgo, 1), 2 ) ...
        );
    g = g + zetaA*theta;
end
end