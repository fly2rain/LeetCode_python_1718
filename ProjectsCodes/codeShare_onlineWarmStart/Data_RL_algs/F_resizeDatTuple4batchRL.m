function [aaa, rrr, X, Y] = F_resizeDatTuple4batchRL (GrOt, GrAt, GrRt)
% Breif:
%       resize the data for the batch reinforcement learning 
%
% Input parameters ################################################
% GrOt   ... (Lo x TT x NN) the observed states at every time point for each people
% GrAt   ... (TT x NN) the observed actions at each time point for each people
% GrRt   ... (TT x NN) tbe immediate reward at each time point for each people
% Output parameters ################################################
% aaa    ... (NSmp) the action of N people at T time points
% rrr    ... (NSmp) the immediate reward of N people at T time points
% X      ... (Lo x NSmp) the set of current states
% Y      ... (Lo x NSmp) the set of next states
%
% References:
%     [1]
% 
% version 1.0 -- 04/12/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% get & set the dimensions ------------------------------------------
[Lo, T, N] = size (GrOt);
NSmp = (T-1) * N;
% creat the matrices to store the data for batch RL -----------------
X   = zeros (Lo, NSmp); % x(O,A): the set of the constructed feature
Y   = zeros (Lo, NSmp);   % Y:  the next states 
rrr = zeros (NSmp, 1);    % rr: the set of immediate reward.
aaa = zeros (NSmp, 1);

subIdxs = 1 : N;
for t = 1 : T - 1
    idxRange = subIdxs + (t-1)*N; 
    % select samples and construct features 
    aaa (idxRange) = GrAt (t, :);
    rrr (idxRange) = GrRt (t, :);
    X (:,idxRange) = reshape( GrOt(:,t,:),   [Lo, N] );   
    Y (:,idxRange) = reshape( GrOt(:,t+1,:), [Lo, N] );
end
end