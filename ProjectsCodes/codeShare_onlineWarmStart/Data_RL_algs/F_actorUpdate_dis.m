% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [f, g] = F_actorUpdate_dis (theta, uuf, w, XG, XX0, XX1, zetaA, Lgo)
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

%  weighted actor updates -----------------------------
f  = - mean ( ((Q_A0.*Pi_0) + (Q_A1.*Pi_1)) .* uuf, 2 ); % min 2 max
f  = f + (zetaA/2)*(theta' * theta);

if nargout > 1 % the gradient g is required
    % #2 d\pi/d\theta ---------------
    dPi_0 = - (expThetaGo .* Pi_0.^2);  % 1 x t
    dPi_1 = - dPi_0;
    
    g = - mean ( XG .* repmat( (Q_A0.*dPi_0)+(Q_A1.*dPi_1), Lgo, 1), 2 );
    g = g + zetaA*theta;
end
end