function [U, V, objErrs] = gnmf_sc_Multi(X, varargin)
% -- [U, V, objErrs] = gnmf_sc_Multi(X, varargin)

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
% X = U*V'
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
%
%   version 1.0 -- 2012-4-6
%
%   Written by Zhu Feiyun

% [errTol, maxIter, lambda, normType, normW, NormWei, U0, V0, wghtTp, dims, winSize, sigma, verb] = ...
%     parse_opt(varargin, 'errTol', 0, 'maxIter', 200, 'lambda', 1, 'normType', 1, ...
%     'normW_orNot', 1, 'NormWeight',  0, 'W0', [], 'H0^T', [], 'weightType', 1, ...
%     'dims', [], 'winSize', 1, 'sigma', 0.5, 'verb', 1);

% alpha  ... normlization parameter of Sparse Coding
% lambda ... normlization parameter of Graph Regularized
[errTol, maxIter, lambda, alpha, weightType, sigma, winSize, normType, normW, verb, ...
    dims, U, V, epss, IDX, percent] = parse_opt(varargin, ...
               'errTol', 1e-4, 'maxIter', 200, 'lambda', 1, 'alpha', 0.5, ...
               'weightType', 1, 'sigma', 0.5, 'winSize', 1, 'normType', 1, ...
               'normW_orNot', 1, 'verb', 10, 'dimOfData', [], ...
               'W0', [], 'H0^T', [] , 'myeps', [], 'IDX', [], ...
               'percent', []);
           
           nSmp = size(X, 2);

% ----------------------------------------------------------------------
% ========================== weigth matrix =============================
if lambda > 0 %  when lambda > 0，need to calculate weight matrix   
    switch weightType
         case 1 % 0-1 weighting
            [ri, ci, val] = mex_weightMatrix_01( dims(1:2), winSize );
        case 2 % Dot-Product Weighting
            [ri, ci, val] = mex_weightMatrix_SAD(reshape(X.', dims), winSize );
        case 3 % heat kernel weighting
            [ri, ci, val] = mex_weightMatrix_gauss(reshape(X.', dims), winSize, sigma);
        case 4 % sad neighbour percent
            [ri, ci, val] = mexWeightSad_neighbour(X, dims, winSize, percent);
        case 5 % corr neighbour percent
            [ri, ci, val] = mexWeightCorr_neighbour(X, dims, winSize, percent);            
        case 6 % sad cluster percent
            [ri, ci, val] = mexWeightSad_cluster(X, dims, IDX, winSize, percent); 
        case 7 % corr cluster percent
            [ri, ci, val] = mexWeightCorr_cluster(X, dims, IDX, winSize, percent);
        otherwise
            [ri, ci, val] = mex_weightMatrix_01( dims(1:2), winSize);
            s = message('weightType must be 1, 2 or 3');
            warning(s);
    end
    W = sparse(ri+1, ci+1, val, nSmp, nSmp);    
    W = lambda*W;
    DCol = full(sum(W, 2));
    D = spdiags(DCol,0,nSmp,nSmp);
%     D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
%     L = D - W;
    % 对L的 normlization
    %     if exist('NormWei', 'var') && NormWei
    %         D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
    %         L = D_mhalf*L*D_mhalf;
    %     end
end
% ======================== weigth matrix =============================
% Initialize the U,H matrix
[U, V] = NormalizeUV(U, V, ~normW, normType);

% calculate errors
objErrs = zeros(maxIter, 1); % objective funtion errors

for t = 1 : maxIter
    % ===================== update V ========================
    XU = X'*U;  % mnk or pk (p<<mn)
    UU = U'*U;  % mk^2
    VUU = V*UU; % nk^2    
    if lambda > 0
        WV = W*V;
        DV = D*V;
        
        XU = XU + WV;
        VUU = VUU + DV + alpha; % normlize 
    end
    
    V = V.*(XU./max(VUU, epss));
    
    % ===================== update U ========================
    XV = X*V;   % mnk or pk (p<<mn)
    VV = V'*V;  % nk^2
    UVV = U*VV; % mk^2
    
    U = U.*(XV./max(UVV, epss)); % 3mk
      
    if(rem(t, 10) == 0)
        [U, V] = NormalizeUV(U, V, ~normW, normType);
    end
    
    % ======= calculate the errors ==========================
    eucErrs  = sum(sum((U*V'-X).^2));
    if lambda > 0
        regErrs1 = sum(sum(V.*(DV-WV)));
    else
        regErrs1 = 0;
    end
    
    regErrs2 = alpha * sum(V(:));
    objErrs(t) = eucErrs + regErrs1 + regErrs2;
    
    if(verb && rem(t, verb) == 0)
        relErrs = ((objErrs(t-verb+1) - objErrs(t)) / objErrs(t-verb+1));
        relErrs = relErrs / (verb - 1);
        fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f  relErrs: %f\n', ...
            t, eucErrs, regErrs1, regErrs2, objErrs(t), relErrs);
        if(abs(relErrs) < errTol)  % check for convergence if asked
            break;
        end
    end
    %     if(t ~= 1)
    %         relErrs = ((objErrs(t-1) - objErrs(t)) / max(objErrs(t-1), epss));
    %         if(abs(relErrs) < errTol)  % check for convergence if asked
    %             break;
    %         end
    %     else
    %         relErrs  = 0;
    %     end
    %
    %     if(verb && rem(t, verb) == 0)
    %         fprintf('[%d] : eucErrs: %f  regErr1: %f  regErr2: %f  objErrs: %f  relErrs: %f\n', ...
    %             t, eucErrs, regErrs1, regErrs2, objErrs(t), relErrs);
    %     end
end
[U, V] = NormalizeUV(U, V, ~normW, normType);
objErrs = objErrs(1:verb:maxIter);
end

% =============================================
function [U, V] = NormalizeUV(U, V, NormV, NormType)
K = size(U,2);
if NormType == 2
    if NormV
        norms = max(1e-15,sqrt(sum(V.^2,1)))';
        V = V*spdiags(norms.^-1,0,K,K);
        U = U*spdiags(norms,0,K,K);
    else
        norms = max(1e-15,sqrt(sum(U.^2,1)))';
        U = U*spdiags(norms.^-1,0,K,K);
        V = V*spdiags(norms,0,K,K);
    end
    
else if NormType == 1
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end
end
end





