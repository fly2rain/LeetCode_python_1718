function [AtN, piN] = F_simuAction (GotN, thetaN)
% Breif:
%    simulate the action for all the N samples, given parameter \theta and the 
%    constructed state features g(o_t). Note that, this program is only suitable for
%    the binary sampling situation.
%
% Input parameters ################################################
% GotN  ... (Lgo x N) the updated feature for the policy function.
% thetaN... (Lgo x N) the parameter in the policy function for all the n samples
%
% Output parameters ################################################
% AtN   ... (1 x N) N input actions, each is action is either 0 or 1.
% piN   ... (1 x N) the probability for all N samples
%
% References:
%     [1] 
%
%  version 1.0 - 10/15/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

N = size (GotN, 2);
% probability of action 0.
piN = 1 ./ ( 1 + exp(sum(thetaN.*GotN, 1)) ); 

rndVal = rand (1, N);

% draw action At according to current policy function
AtN = zeros (1, N);
AtN ( rndVal > piN ) = 1;

if nargout == 2
    piN = horzcat (piN, 1 - piN);
end
end