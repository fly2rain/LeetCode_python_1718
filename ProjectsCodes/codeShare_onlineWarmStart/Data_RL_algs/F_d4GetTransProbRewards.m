function [GrPr, GrR] = F_d4GetTransProbRewards (Lov, datIdx, alpha, La)
% Breif:
%     get the transition probability & reward function on data #4, 
%     for the use of the mdp_relative_value_iteration
%
% Input parameters ################################################
% Lov 	 ... (1 x 1) the number of value choices for each of the element
% datIdx ... (1 x 1) the simulation data #
% alpha  ... (1 x 1) the parameter to balance the problem more like pure bnadit or Reinforment leanring
% La     ... (1 x 1) the number of action choice
%             
% Output parameters ################################################
% GrPr ... (SxSXA) the state transition probability.
% GrR  ... (SxSXA) the reward model
% 
%  version 1.0 - 02/11/2016
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 4
    if nargin < 3 
        alpha = 1;
    end
    La = 2;
end

if datIdx ~= 4
    error ('datIdx must be equal to 4.\n');
end

%% 1 Get the state transition probability %%%%%%%%%%%%%%%%
GrPr = zeros (Lov, Lov, La);
Pr0 = [ % the transition matrix of S_t given A_t=0
    0.90,  0.10,   0.00,   0.00,   0.00; ...
    0.80,  0.00,   0.20,   0.00,   0.00; ...
    0.85,  0.00,   0.00,   0.15,   0.00; ...
    0.90,  0.00,   0.00,   0.00,   0.10; ...
    0.95,  0.00,   0.00,   0.00,   0.05  ...
    ];
Pr1 = [ % the transition matrix of S_t given A_t=1
    0.10,  0.90,   0.00,   0.00,   0.00; ...
    0.20,  0.00,   0.80,   0.00,   0.00; ...
    0.15,  0.00,   0.00,   0.85,   0.00; ...
    0.10,  0.00,   0.00,   0.00,   0.90; ...
    0.05,  0.00,   0.00,   0.00,   0.95  ...
    ];
%
Pr_cha = Pr1 - Pr0;
Pr1    = Pr0 + alpha*Pr_cha;
% 
ia=1;   GrPr(:,:,ia) = Pr0;
ia=2;   GrPr(:,:,ia) = Pr1;

%% 2 Get the reward function %%%%%%%%%%%%%%%%%%%%%%%%%%%   
R = [ % the transition matrix of S_t given A_t=0
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    0.01,  0.00,   0.00,   0.00,   0.00; ...
    1.10,  0.00,   0.00,   0.00,   0.90  ...
    ];
GrR = repmat ( R, [1,1,2] );