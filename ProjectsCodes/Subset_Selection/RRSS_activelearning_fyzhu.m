function [rank_idx, rank_value, A, obj] = RRSS_activelearning_fyzhu (X, r, NIter)
% XX: if iskernel = 0, XX is a d*n data matrix, each column is a data point
%     if iskernel ~= 0, XX is a n*n kernel matrix
% r: parameter, larger to select fewer data points
% iskernel: kernel version if iskernel ~= 0, and XX should be kernel matrix
% rank_idx: the ranking values of data points
% rank_value: the ranking index of data points
% A: n*n representation matrix
% obj: objective values in the iterations

% Ref:
% Feiping Nie, Hua Wang, Heng Huang, Chris Ding.
% Early Active Learning via Robust Representation and Structured Sparsity.
% The 23rd International Joint Conference on Artificial Intelligence (IJCAI), 2013.
if nargin < 3
    NIter = 15;
end

[A, obj]=L12R21(X, r, NIter);

a = sum(abs(A), 2);
[rank_value, rank_idx] = sort(a,'descend');


% $$ min_X  ||(XA - X)^t||_{2,1} + r * ||X||_{2,1} $$
function [A, obj]=L12R21(X, r, NIter, A0)

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

if N <= L
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
else % N > L
    for iter = 1:NIter
        Vinv  = spdiags (2.*vv, 0, N, N);
        XVinv = ( X * Vinv ).';
        for n = 1 : N
            un = u_inv (n);
            xn = X (:, n);
            A(:, n) = ( un .* XVinv ) * ( ( un*X*XVinv + r*Ifea ) \ xn );
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