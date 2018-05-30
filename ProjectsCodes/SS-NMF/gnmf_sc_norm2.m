function [U, V, objErrs] = gnmf_sc_norm2(X, U, V, W, varargin)
% -- function [U, V, objErrs] = gnmf_sc_norm2(X, U, V, W, varargin)
% Graph regularized Non-negative Matrix Factorization (GNMF) with
%          multiplicative update
%
% where
%   X
% Notation:
% X ... (nBand x nSmp) data matrix
%       nBand ... number of spectral bands 
%       nSmp  ... number of pixels
% W ... weight matrix of the affinity graph
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
% 8, pp. 1548-1560, 2011.
%
%   version 2.0 -- 2012-7-18
%
%   Written by Zhu Feiyun
[errTol, maxIter, lambda, alpha, normType, normW, verb, epss, fIter] = parse_opt(varargin, ...
         'errTol', 1e-4, 'maxIter', 200, 'lambda', 1, 'alpha', 0.5, ...
         'normType', 2, 'normW', 1, 'verb', 10, 'myeps', [], 'firstIterate', 0);
     
     nSmp = size(X, 2);
     if lambda > 0
         W = lambda * W;
         DCol = full(sum(W, 2));
         D = spdiags(DCol,0,nSmp,nSmp);
     end

% ======================== weigth matrix =============================

% calculate errors
objErrs = zeros(maxIter, 1); % objective funtion errors

% * first iterate V based good U ***
UX = U' * X;  % mnk or pk (p<<mn)
UU = U' * U;  % mk^2
for t = 1 : fIter
    UUV = UU * V; % nk^2    
    V = V.*(UX./max(UUV, epss));
end
% ********************************
% Initialize the U,H matrix
[U, V] = normalize_WH(U, V, normType, normW);

for t = 1 : maxIter
    % ===================== update V ========================
    UX = U' * X;  % mnk or pk (p<<mn)
    UU = U' * U;  % mk^2
    UUV = UU * V; % nk^2
    
    if lambda > 0
        VW = V*W;
        VD = V*D;
        
        UX = UX + VW;
        UUV = UUV + VD; % normlize
    end
    UUV = UUV + alpha;
    
    V = V.*(UX./max(UUV, epss));
    
    % ===================== update U ========================
    XV = X*V';   % mnk or pk (p<<mn)
    VV = V*V';  % nk^2
    UVV = U*VV; % mk^2
    
    U = U.*(XV./max(UVV, epss)); % 3mk
      
    if(rem(t, verb) == 0)
        [U, V] = normalize_WH(U, V, normType, normW);
    end
    
    % ======= calculate the errors ==========================
    eucErrs  = sum(sum((U*V-X).^2));
    if lambda > 0
        regErrs1 = sum(sum(V.*(VD-VW)));
    else
        regErrs1 = 0;
    end
    regErrs2 = alpha * sum(V(:));
    objErrs(t) = eucErrs + regErrs1 + regErrs2;
    if t == 1
        fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f\n', ...
            t, eucErrs, regErrs1, regErrs2, objErrs(t));
    end
    if(verb && rem(t, verb) == 0)
        relErrs = ((objErrs(t-verb+1) - objErrs(t)) / objErrs(t-verb+1));
        relErrs = relErrs / (verb - 1);
        fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f  relErrs: %f\n', ...
            t, eucErrs, regErrs1, regErrs2, objErrs(t), relErrs);
        if(abs(relErrs) < errTol)  % check for convergence if asked
            break;
        end
    end
end
[U, V] = normalize_WH(U, V, normType, normW);
objErrs = objErrs(1:verb:maxIter);
end