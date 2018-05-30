function [U, V, objErrs] = gnmf_sc_norm3(X, U, V, W, varargin)
% -- [U, V, objErrs] = gnmf_sc_norm3(X, U, V, W, varargin)

% Graph regularized Non-negative Matrix Factorization (GNMF) with
%          multiplicative update
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% W ... weight matrix of the affinity graph
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*V
%
% References:
% [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
% Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
% (ICDM'08), Pisa, Italy, Dec. 2008.
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
% 8, pp. 1548-1560, 2011.%
%
%   version 1.0 -- 2012-7-17
%
%   Written by Zhu Feiyun


% alpha  ... normlization parameter of Sparse Coding
% lambda ... normlization parameter of Graph Regularized
[errTol, maxIter, lambda, alpha, delta, verb, epss] = parse_opt(varargin, ...
               'errTol', 1e-4, 'maxIter', 200, 'lambda', 1, 'alpha', 0.5, ...
               'delta', 2, 'verb', 10, 'myeps', []);
           
[nBand, nEnd] = size(U);
nSmp = size(X, 2);

W = lambda*W;
DCol = full(sum(W, 2));
D = spdiags(DCol,0,nSmp,nSmp);

% calculate errors
objErrs = zeros(maxIter, 1); % objective funtion errors
Xa = [X; delta*ones(1, nSmp)];

for t = 1 : maxIter
    % ===================== update V ========================
    Ua = [U; delta*ones(1, nEnd)];
    UX = Ua' * Xa;  % mnk or pk (p<<mn)
    UU = Ua'* Ua;  % mk^2
    UUV = UU * V; % nk^2    
%     if lambda > 0
        VW = V*W;
        VD = V*D;
        
        UX = UX + VW;
        UUV = UUV + VD + alpha; % normlize 
%     end
    V = V.*(UX./ max(UUV, epss));
    
    % ===================== update U ========================
    XV = X*V';   % mnk or pk (p<<mn)
    VV = V*V';  % nk^2
    UVV = U*VV; % mk^2
    
    U = U.*(XV./ max(UVV, epss)); % 3mk
    
    % ======= calculate the errors ==========================
    eucErrs  = sum(sum((U*V-X).^2));
    regErrs1 = sum(sum( (VD-VW) .* V ));
    
    regErrs2 = alpha * sum(V(:));
    objErrs(t) = eucErrs + regErrs1 + regErrs2;
    
    if(t == 1)
        fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f\n', ...
            t, eucErrs, regErrs1, regErrs2, objErrs(t));
    end
    
    if(verb && rem(t, verb) == 0 )
        relErrs = ((objErrs(t-verb+1) - objErrs(t)) / objErrs(t-verb+1));
        relErrs = relErrs / (verb - 1);
        fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f  relErrs: %f\n', ...
            t, eucErrs, regErrs1, regErrs2, objErrs(t), relErrs);
        if(abs(relErrs) < errTol)  % check for convergence if asked
            break;
        end
    end
end
objErrs = objErrs(1:verb:maxIter);