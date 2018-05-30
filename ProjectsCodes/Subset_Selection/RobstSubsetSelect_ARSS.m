function [slctIdx, slctVal, A, objErr] = RobstSubsetSelect_ARSS (X, varargin)
% Breif:
%      10,000+ Times Accelerated Robust Subset Selection (ARSS)
%
% Objective function:
%      min_A  ||X - XA||_p^p + r ||A||_2,1
%
% Notation: L
% X ... (L x N) data matrix, where L is the number of features, and
%           N is the number of samples.
% A ... (N x N) is a row structured sparse matrix used to select
%           the most representive and informative samples.
% p, r are nonnegative parameters given beforehand.
%

% References:
%     [1] Feiyun Zhu, Bin Fan, Ying Wang, Shiming Xiang and Chunhong Pan.
%     "10,000+ Times Accelerated Robust Subset Selection (ARSS)". (AAAI,
%     under review), CORR, abs/1409.3660, 2014.
%
%  version 2.0 -- 2014-10-21
%  version 1.0 -- 2014-07-21
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

[r, maxIter, mu, rho, p, verb] = parse_opt (varargin, ...
    'r', 0.1, 'maxIter', 100, 'mu', 0.001, 'rho', 1.1, 'p', 0.5, 'verb', 5);

[A, objErr] = Lp_R21 (X, r, maxIter, p, mu, rho, verb);

% remove the rows that construct others sparsely.
a = sum (abs(A), 2);
sprsVal = get_SprsMapColumn ( A.' );
a ( sprsVal > 0.8 ) = 0;
[slctVal, slctIdx] = sort (a, 'descend');
end


% ===========================================================
function [A, rErr] = Lp_R21 (X, r, maxIter, p, mu, rho, verb)

LpGIS    = @(C, lambda, p) solve_Lp (C, lambda, p);
L1Shrink = @(C, lambda) solve_L1 (C, lambda);
L2Shrink = @(C, lambda) solve_L2 (C, lambda);

if p < 0
    p = 0.5;
end

[L, N] = size (X);

Lambda = zeros (L, N);
A = eye (N); XA = X;
% A = rand (N); XA = X * A;

Ifea = speye (L, L);

t = maxIter;
first_linearSystem_orNot = t*(N^3 - L^3) + ( (t+1)*N - 2*t*L - t )*N*L <= 0;
if first_linearSystem_orNot
    XTX = X.'*X;
end

for iter = 1 : maxIter
    % ==== solve B by the Generalized Iterated Shrinkage method ========
    lambda = 1 / mu; % lowercase lambda is a penalty parameter
    H = X - XA - Lambda .* lambda; % uppercase Lambda is the lagrange multiplier
    if p < 1
        B = LpGIS (H, lambda, p);
    elseif p == 1
        B = L1Shrink (H, lambda);
    elseif p == 2
        for n = 1 : nSmp % robust to samples, column sparse
            B(:,n) = L2Shrink (H(:,n), lambda);
        end
    end
    
    % ==== solve A ============
    vvv = ( sum ( A.*A, 2 ) + eps ) .^ 0.5;
    beta = mu / r;
    P = X - B - Lambda .* lambda;
    if first_linearSystem_orNot % choose the 1st linear system
        v_inv = 1 ./ vvv;
        V = spdiags (v_inv, 0, nSmp, nSmp);
        A = beta * ( (V + beta*XTX) \ (X.' * P) );
    else % choose the 2nd linear system
        Vinv = spdiags (vvv, 0, nSmp, nSmp);
        X_Vinv = (X * Vinv).';
        A = beta * X_Vinv * ( ( Ifea + beta .* (X * X_Vinv) ) \ P );
    end
    
    XA = X * A;
    Lambda = Lambda + mu .* ( B - X + XA);
    mu = min(10^4, mu*rho);
    
    % ===== if calculate the objective error ==============
    if ( verb && rem(iter, verb) == 0 )
        rErr = sum (sum( abs( X - XA ) .^ p ));
        objErr = rErr + r * sum(vvv);
        fprintf('[%d] : rErr: %f objErr: %f \n', iter, rErr, objErr);
    else
        rErr = 0;
    end
end
end

% ===========================================================
% min_y 0.5 (y-c)^2 + r * |y|^p
%  where y and c are vectors or matrices
function   y = solve_Lp ( c, r, p )
% Modified by Dr. Weisheng Dong
J     =  2;
tau   =  (2*r.*(1-p))^(1/(2-p)) + p*r.*(2*(1-p)*r)^((p-1)/(2-p));
y     =  zeros( size(c) );
i0    =  find( abs(c)>tau );

if length(i0) >= 1
    % lambda  =   lambda(i0);
    c0    =   c(i0);
    t     =   abs(c0);
    for  j  =  1 : J
        t    =  abs(c0) - p*r.*(t).^(p-1);
    end
    y(i0)   =  sign(c0).*t;
end
end

% ===========================================================
% min_y 0.5 (y-c)^2 + r * |y|
%  where y and c are vectors or matrices
function   y = solve_L1 ( c, r )
% author: feiyun Zhu
y = sign(c) .* max(abs(c) - r, 0);
end

% ===========================================================
% min_y 0.5 (y-c)^2 + r * |y|_2
%   where y and c are vectors..
function   y = solve_L2 ( c, r )
% author: feiyun Zhu
cNorm = norm (c) + eps;
y = c .* ( max(cNorm - r, 0) ./ cNorm );
end