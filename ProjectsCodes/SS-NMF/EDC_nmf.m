function [M, A, objErrs] = EDC_nmf (Y, M, A, varargin)
% -- function [U, V, objErrs] = EDC_nmf (X, U, V, W, varargin)
% 
% breif:    
%       Endmember Dissimilarity Constrained NMF (EDCNMF).
%
% where
%   Y = M*A
% Notation:
% Y ... (nBand x nSmp) data matrix
%       nBand ... number of spectral bands 
%       nSmp  ... number of pixels
% W ... weight matrix of the affinity graph
%
%
% References:
% [1] "An Endmember Dissimilarity Constrained Non-Negative Matrix Factorization Method
%               for Hyperspectral Unmixing"
%
%  version 1.0 -- 2013-7-18
%
%  Written by Zhu Feiyun

[errTol, maxIter, lambda, normType, normW, verb, epss, fIter] = parse_opt (varargin, ...
         'errTol', 1e-4, 'maxIter', 200, 'lambda', 1, 'normType', 2, ...
         'normW', 1, 'verb', 10, 'myeps', 1e-10, 'firstIterate', 0);
      
% Construct two auxiliary matrixes
[nBand, nEnd] = size (M);

H = cat (1, -1*eye(nBand-1), zeros(1, nBand-1)) + ...
    cat (1, zeros(1, nBand-1), eye(nBand-1));

T = nEnd * eye (nEnd) - ones (nEnd);

% calculate errors
objErrs = zeros(maxIter, 1); % objective funtion errors

% * first iterate A based good M ***
MY = M' * Y;  % mnk or pk (p<<mn)
MM = M' * M;  % mk^2
for iter = 1 : fIter
    MMA = MM * A; % nk^2    
    A = A.*(MY./max(MMA, epss));
end

% ********************************
% Initialize the M,A matrix
[M, A] = normalize_WH(M, A, normType, normW);

for iter = 1 : maxIter
    % ===================== update A ========================
    MY = M' * Y;  % KLN
    MM = M' * M;  % kL^2
    MMA = MM * A; % NK^2
    
    A = A.*(MY./max(MMA, epss));    
    A = max (A, epss);
    
    % ===================== update M ========================
    YA = Y*A.';  % KLN (p<<mn)
    AA = A*A.';  % NK^2
    MAA = M*AA; % LK^2
    
    HM = H.' * M;
    
    HHMT = H * HM * T;
    M = M.*((YA - lambda*HHMT)./max(MAA, epss)); % 3mk
    
    M = max (M, epss);
      
    if(rem(iter, verb) == 0)
        [M, A] = normalize_WH(M, A, normType, normW);
    end
    
    % ======= calculate the errors ==========================
    eucErrs  = 0.5 * sum(sum((M*A-Y).^2));
    
    regErrs = 0.5*lambda* trace(HM * T * HM.');
 
    objErrs(iter) = eucErrs + regErrs;
    
    if iter == 1
        fprintf('[%d] : eucErrs: %f  regErr: %f  objErrs: %f\n', ...
            iter, eucErrs, regErrs, objErrs(iter));
    end
    
    if(verb && rem(iter, verb) == 0)
        relErrs = ((objErrs(iter-verb+1) - objErrs(iter)) / objErrs(iter-verb+1));
        relErrs = relErrs / (verb - 1);
        
        fprintf('[%d] : eucErrs: %f  regErr: %f  objErrs: %f  relErrs: %f\n', ...
            iter, eucErrs, regErrs, objErrs(iter), relErrs);
        
        if(abs(relErrs) < errTol)  % check for convergence if asked
            break;
        end
    end
end

[M, A] = normalize_WH(M, A, normType, normW);
% objErrs = objErrs(1:verb:maxIter);
end