function [W, H, objErrs, iter] = locRepreNmf_Multi(V, r, varargin)
%  [W, H, objErrs, iter] = locRepreNmf_Multi(V, r, varargin)
% Graph regularized Non-negative Matrix Factorization (GNMF) with
%          multiplicative update
%
% where
%   V
% Notation:
% V ... (mFea x nSmp) data matrix
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% L ... graph Laplace
%
% varargin ... Cell holding all settings
%
% You only need to provide the above four inputs.
%
% V = W*H
%     W --- Dictionary (nFea x r)
%     H --- Coding (r x nSmp)
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
%   version 1.0 -- 2012-4-6
%
%   Written by Zhu Feiyun
%
%
% --------------- default settings --------------------------
[errTol, maxIter, lambda, weightType, sigma, winSize, normType, normW, verb, ...
    dims, U, V, epss] = parse_opt(varargin, 'errTol', 0, 'maxIter', 200, 'lambda', 1,  ...
               'weightType', 1, 'sigma', 0.5, 'winSize', 1, 'normType', 1, ...
               'normW_orNot', 1, 'verb', 1, 'dimOfData', [], 'W0', [], 'H0^T', [] ,...
               'myeps', 1e-20);
           nFea = dim(3);
           nSmp = dims(1) * dims(2);
% verb: if verb printf;
%       else  non printf;
% ------------------------------------------------------------

[mFea,nSmp]=size(V);

% ----------------------------------------------------------------------
% ========================== weigth matrix =============================
if lambda > 0 %  when lambda > 0，need to calculate weight matrix    
    [ri, ci, val] = mex_localLinearRre(reshape(V.', dims),  ws, sigma);    
    L = sparse(ri+1, ci+1, val);   
    
% %     DCol = full(sum(W,2));
% %     D = spdiags(DCol,0,nSmp,nSmp);
% %     L = D - W;
% %     if exist('NormWei', 'var') && NormWei
% %         D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
% %         L = D_mhalf*L*D_mhalf;
% %     end
end
% ======================== weigth matrix =============================

if lambda < 0
    error( 'the punishment parameter must be positive.\n' );
end

% Initialize the U,H matrix
if isempty(W0)
    W = rand(mFea,r);
else
    W = max(W0, epss);
end

if isempty(H0)
    H = rand(r, nSmp);
else
    H = max(H0, epss);
end

[W, H] = normalize_WH(W, H, normType, normW);

% calculate errors
eucErrs = zeros(maxIter, 1); % euc errors
regErrs = zeros(maxIter, 1); % regularization errors
objErrs = zeros(maxIter, 1); % objective funtion errors
relErrs = zeros(maxIter, 1); % the relateive errors  of the objErrs

eucErrs(1)  = norm(V - W*H, 'fro')^2;
if lambda > 0
    regErrs(1) = sum(sum((H*L).* H));
else
    regErrs(1) = 0;
end
objErrs(1) = eucErrs(1) + lambda*regErrs(1);

for iter = 2 : maxIter
    % ===================== update H ========================
    WV = W' * V;  % rnm (r<<min(m,n))
    WW = W' * W;  % nr^2
    WWH = WW * H; % mr^2
    if lambda > 0
        HL = H * L;   % rm^2
        WWH = WWH + lambda*HL;       
    end
    H = H.*(WV./max(WWH, epss));
    %   H = H.*(XH./max(VUU, epss));    
    if(rem(iter, 10) == 0)
        [W, H] = normalize_WH(W, H, normType, normW);
    end
    % ===================== update W ========================
    VH = V*H';  % rnm (r<<min(m,n))
    HH = H*H';  % mr^2
    WHH = W*HH; % nk^2
    W = W.*(VH./max(WHH, epss)); % 3nr
    
    % ======= calculate the errors ==========================
    eucErrs(iter)  = norm(V - W*H, 'fro')^2;
    if lambda > 0
        regErrs(iter) = sum(sum((H*L).* H));
    else
        regErrs(iter) = 0;
    end
    objErrs(iter) = eucErrs(iter) + lambda*regErrs(iter);
    relErrs(iter) = abs( ( objErrs(iter) - objErrs(iter-1) ) / max(objErrs(iter-1), epss) );
    
    if( relErrs(iter) < errTol )
        break;
    end
    
    if(verb && rem(iter, 10) == 0)
        fprintf('[%d]:  eucErrs:%.4f  objErrs:%.4f  relErrs:%.8f\n', ...
            iter, eucErrs(iter), objErrs(iter), relErrs(iter));
    end
end
[W, H] = normalize_WH(W, H, normType, normW);
end