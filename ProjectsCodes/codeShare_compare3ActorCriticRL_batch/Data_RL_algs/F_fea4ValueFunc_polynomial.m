function Y = F_fea4ValueFunc_polynomial (X, order)
% Breif:
%   construct the n-order polynomial feature simultaneously for N input samples
%
% Input parameters ################################################
% X     ... (L x N) contains N samples; each sample has L feature elements.
% order ... (1 x 1) the order value of the polynomial basis
%
% Output parameters ################################################
% Y     ... (Lf x N) the constructed feature vector
%
% References:
%     [1]
%
% version 1.0 -- 01/11/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)
if order == 0
    Y = X;
else
    [Lo, N] = size (X); % X contains N samples; each sample has L feature elements.
    Lfo     = (order+1) ^ Lo; % the length of the constructed polynomial feature
    C       = zeros (Lo, Lfo); % the order matrix with L x (n+1)^L elements
    idxSequence = 0 : Lfo-1;  % the sequence to construct the order matrix C effectively.
    
    % get the order matrix for the polynomial basis
    denom = (order+1) ^ (Lo-1);
    for i = 1 : Lo-1
        C(Lo-i+1,:) = fix ( idxSequence ./ denom );
        idxSequence = rem ( idxSequence, denom );
        denom       = denom / (order+1);
    end
    C(1,:) = idxSequence;
    
    % get the constructed polynomial feature
    Y = zeros (Lfo, N);
    for n = 1 : N
        xn = X (:, n);
        Y(:,n) = prod ( repmat(xn,1,Lfo) .^ C, 1 ).';
    end
end