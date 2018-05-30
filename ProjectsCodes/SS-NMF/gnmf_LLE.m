function [W, H, objErrs] = gnmf_LLE(V, varargin)
% - [W, H, objErrs] = gnmf_LLE(V, varargin)
% Local Linear Embedding regularized Non-negative Matrix Factorization 
%    (GNMF) with multiplicative update in Eucler Space
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
%   version 2.0 -- 2012-5-23
%
%   Written by Zhu Feiyun
%
%
% --------------- default settings --------------------------
[errTol, maxIter, lambda, weightType, epsilon, winSize, normType, normW, verb, ...
    dims, W, H, epss] = parse_opt(varargin, 'errTol', 0, 'maxIter', 200, 'lambda', 1,  ...
               'weightType', 1, 'epsilon', 0.5, 'winSize', 1, 'normType', 1, ...
               'normW_orNot', 1, 'verb', 1, 'dimOfData', [], 'W0', [], 'H0', [] ,...
               'epss', 1e-20);
% verb: if verb printf;
%       else  non printf;
% epsilon: LLe regularized parameter, a small number by default.
% ------------------------------------------------------------

nSmp = size(V, 2);
% ----------------------------------------------------------------------
% ========================== weigth matrix =============================
if lambda > 0 %  when lambda > 0，need to calculate weight matrix 
    % h = Theta v + b, b是局部线性的截距
    % 在求解Theta和b的时候，四种方法都考虑对Theta的2范正则了。
    %   求解重构误差的时候11，21方法没有考虑正则；12，22考虑了正则。
    switch weightType
        case 11 % no b, no regularized parameter Theta
            [ri, ci, val] = mexLocLinearEmbed1_1(reshape(V.', dims),  winSize, epsilon);
        case 12 % no b, have regularized parameter Theta
            [ri, ci, val] = mexLocLinearEmbed1_2(reshape(V.', dims),  winSize, epsilon);
        case 21 % have b, no regularized parameter Theta
            [ri, ci, val] = mexLocLinearEmbed2_1(reshape(V.', dims),  winSize, epsilon);
        case 22 % have b, have regularized parameter Theta
            [ri, ci, val] = mexLTSA2(reshape(V.', dims),  winSize, epsilon);
        otherwise
            [ri, ci, val] = mexLocLinearEmbed2_2(reshape(V.', dims),  winSize, epsilon);
            s = message('weightType must be 11, 12, 21 or 22');
            warning(s);
    end
    L = sparse(ri+1, ci+1, val);   
    L = lambda*L;
    R = L;  R(R<0) = 0;
    P = -L; P(P<0) = 0;
end
% ======================== weigth matrix =============================

% Initialize the U,H matrix
[W, H] = normalize_WH(W, H, normType, normW);

% objective errors
objErrs = zeros(maxIter, 1); % objective funtion errors

for t = 1 : maxIter
    % ===================== update H ========================
    WV = W' * V;  % O(rnm) (r<<min(m,n))--r*m matrix, the same size as H matrix
    WW = W' * W;  % O(nr^2) -- r*r matrix
    WWH = WW * H; % O(mr^2) -- r*m matrix, the same size as H matrix
    if lambda > 0
        HR = H * R;   % rm^2
        HP = H * P;
        
        WV = WV + HP;
        WWH = WWH + HR;       
    end
    H = H.*(WV./max(WWH, epss)); % 2rm 

    % ===================== update W ========================
    VH = V*H';  % rnm (r<<min(m,n))
    HH = H*H';  % mr^2 -- r*r matrix
    WHH = W*HH; % nr^2 -- n*r matrix 
    W = W.*(VH./max(WHH, epss)); % 2nr
    
    if(rem(t, 10) == 0)
        [W, H] = normalize_WH(W, H, normType, normW);
    end
    
    % ======= calculate the errors ==========================
    eucErrs  = sum(sum((V - W*H).^2));
    if lambda > 0
        regErrs = sum(sum((HR-HP).*H));
    else
        regErrs = 0;
    end
    objErrs(t) = eucErrs + regErrs;
       
    if(verb && rem(t, verb) == 0)
        relErrs = ((objErrs(t-verb+1) - objErrs(t)) / objErrs(t-verb+1));
        relErrs = relErrs / (verb - 1);
        fprintf('[%d] : eucErrs: %f  regErr: %f  objErrs: %f  relErrs: %f\n', ...
            t, eucErrs, regErrs, objErrs(t), relErrs);
        if(abs(relErrs) < errTol)  % check for convergence if asked
            break;
        end
    end
end
[W, H] = normalize_WH(W, H, normType, ~normW);
objErrs = objErrs(1:verb:maxIter);
end