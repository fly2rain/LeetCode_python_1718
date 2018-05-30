function [rank_idx, rank_value, A, obj] = RRSS_speedup (X, r, NIter)
% Breif:
%       Speedup Robust Sample Selection Algorithm via Self Representation.
%
% Objective function:
%       min_{A}  ||X-XA||_{1,2} + r||A||_{2,1}
%
% Notation:
% X     ... (L x N), where 
%           L ... feature length
%           N ... data size
% r     ... scalar: nonnegative balancing parameter.
% NIter ... the number of iteration steps.
%

% References:
% [1] Feiping Nie, Hua Wang, Heng Huang, Chris Ding. "Early Active 
% Learning via Robust Representation and Structured Sparsity", IJCAI2013.
%
% [2] Feiyun Zhu, Bin Fan, Shiming Xiang. "10,000+ Times Accelerated
% Robust Subset Selection (ARSS)", arXiv:1409.3660.
%
%  version 1.0 -- 2014-7-22
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

if nargin < 3
    NIter = 15;
end

[A, obj] = L12R21_selfRepresentation(X, r, NIter);

a = sum(abs(A), 2);
[rank_value, rank_idx] = sort(a,'descend');


% $$ min_A  ||(XA - X)^T||_{2,1} + r * ||A||_{2,1} $$
function [A, obj] = L12R21_selfRepresentation(X, r, NIter, A0)

verb = 5;
[L, N] = size(X);
A = zeros (N, N);
if nargin < 4
    vv    = ones(N, 1);
    v_inv = 0.5 ./ vv;
    uu    = ones(N, 1);
    u_inv = 0.5 ./ uu;
else
    vv = sqrt( sum(A0 .* A0, 2) + eps );
    v_inv = 0.5 ./ vv;
    
    XA = X*A0 - X;
    uu = sqrt( sum(XA.*XA, 1) + eps );
    u_inv = 0.5 ./ uu;
end;

Ifea = speye (L);
% Ismp = speye (N);

if N <= L % data size is less than or equal to feature length.
    XTX = X.' * X;
    for iter = 1:NIter
        V = spdiags (v_inv, 0, N, N);
        for n = 1 : N
            un = u_inv (n);
            A(:, n) = ( un*XTX + r*V ) \ ( un*XTX(:, n)) ;
        end
        
        vv = sqrt( sum(A.*A, 2) + eps );
        v_inv = 0.5 ./ vv;
        XA = X*A - X;
        uu = sqrt( sum(XA .*XA, 1) + eps );
        u_inv = 0.5 ./ uu;
        
        if ( verb && rem(iter, verb) == 0 )
            rErr = sum(uu);
            obj(iter) = rErr + r*sum(vv);
            fprintf('[%d] : rErr: %f objErr: %f \n', iter, rErr, obj(iter));
        end
    end
    
else % N > L: data size is greater than feature length.
    for iter = 1:NIter
        Vinv  = spdiags (2.*vv, 0, N, N);
        XVinv = ( X * Vinv ).';
        for n = 1 : N
            un = u_inv (n);
            xn = X (:, n);
            A(:, n) = ( un .* XVinv ) * ( ( un.*X*XVinv + r*Ifea ) \ xn );
        end
        
        vv = sqrt( sum(A.*A, 2) + eps );
        v_inv = 0.5 ./ vv;
        XA = X*A - X;
        uu = sqrt( sum(XA .*XA, 1) + eps );
        u_inv = 0.5 ./ uu;
        
        if ( verb && rem(iter, verb) == 0 )
            rErr = sum(uu);
            obj(iter) = rErr + r*sum(vv);
            fprintf('[%d] : rErr: %f objErr: %f \n', iter, rErr, obj(iter));
        end
    end
end