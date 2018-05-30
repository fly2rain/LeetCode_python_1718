function [slctIdx, slctVal, P, Q, objErr] = RSS_JSS (X, varargin)
%
% breif:
%      Co-Selection via Robust Representation and Joint Structured Sparsity
%
% objective function:
%      min_{P,Q}  ||X-X(P+Q)||_p^p + lambda ||P||_2,1 + beta ||Q^T||_2,1
%
% Notation:
% X ... (nFea x nSmp) data matrix, where nFea is the number of features, and
%           nSmp is the number of samples.
% P ... (nFea x nFea) is a column structured sparse matrix used to select
%           the most representive and informative features.
% Q ... (nSmp x nSmp) is a row structured sparse matrix used to select
%           the most representive and informative samples.
%
% p, alpha, and beta are nonnegative parameters given beforehand.
%
%
% References:
%       new idea of mine
%
%  version 1.0 -- 2014-7-22
%
%  Written by Feiyun Zhu

[maxIter, lambda, beta, verb, iskernel, sprsReset] = parse_opt (varargin, ...
    'maxIter', 100, 'lambda', 10, 'beta', 10, 'verb', 5, ...
    'iskernel', 0, 'sprsReset', 5);

if ~iskernel %  X is a L x N kernel matrix
    [P, Q, objErr] = LF_R21R21 (X, maxIter, lambda, beta, verb, sprsReset);
else % X is a N x N kernel matrix
    [P, Q, objErr] = LF_R21R21_kernel (X, maxIter, lambda, beta, verb);
end

% slctVal = 0; slctIdx = 0;
a = sum (abs(P), 2);
[slctVal, slctIdx] = sort (a, 'descend');

b = sum (abs(Q), 1);
rejectSet = find (b > 0.1);

slctIdx = setdiff (slctIdx, rejectSet);
slctVal = slctVal (slctIdx);
end

% ===========================================================
function [P, Q, objErr] = LF_R21R21 (X, maxIter, lambda, beta, verb, sprsReset)
columnSprsMap = @(A) get_SprsMapColumn (A);

[nFea, nSmp] = size (X);

P = zeros (nSmp);
Q = zeros (nSmp);
% P = eye (nSmp);
% Q = eye (nSmp) * 0.4;
uuu = ones (nSmp, 1);
vvv = ones (nSmp, 1);

Ifea = speye (nFea, nFea);
Ismp = speye (nSmp, nSmp);
% minX = min ( X(:) );

if nSmp < nFea % nSmp < nFea =======================
    XTX = X.' * X;
    for iter = 1 : maxIter
        % ==== solve P ====== closed form ======
        u_inv = 0.5 ./ uuu;
        U = spdiags (u_inv, 0, nSmp, nSmp);
        P = ( XTX + lambda.*U ) \ ( XTX * (Ismp - Q) );
        if minX > 0
            P = max (P, 0);
        end
        XP = X - X*P;
        uuu = ( sum ( P.*P, 2 ) + eps ) .^ 0.5;
        
        % ==== solve Q ====== closed form ======
        if beta > 0
            v_inv = 0.5 ./ vvv;
            for n = 1 : nSmp
                vn = v_inv (n);
                hn = XP (:, n);
                Q(:, n) = ( XTX + beta*vn.*Ismp ) \ ( X.' * hn );
            end
            if minX > 0
                Q = max (Q, 0);
            end
            vvv = ( sum ( Q.*Q, 1 ) + eps ) .^ 0.5;           
            
        else % never update 'Q'
            vvv = 0;
            Q = 0;
        end
        
        % ===== if calculate the objective error ==============
        if ( verb && rem(iter, verb) == 0 )
            rErr = sum (sum( abs( XP- X*Q ) .^ 2 ));
            objErr = rErr + lambda * sum( uuu ) + beta * sum ( vvv );
            fprintf('[%d] : rErr: %f objErr: %f \n', iter, rErr, objErr);
        end
    end
    
else % nSmp >= nFea =======================
    XXT = X * X.';
    for iter = 1 : maxIter
        % ==== solve P ====== close form ======        
        u_inv = 2 .* uuu;
        Uinv  = spdiags (u_inv, 0, nSmp, nSmp);
        XUinv = (X * Uinv) .';
        P = XUinv * ( (X * XUinv + lambda * Ifea) \ (X * (Ismp - Q)) );
        %         if minX > 0
        %             P = max (P, 0);
        %         end
        % if the sparsity of the row vectors in P is low, then set them zero
        if ( iter==1 || rem(iter, sprsReset) == 0 )
            sprsVal = columnSprsMap ( P.' );
            rejectIdxs =  sprsVal > 0.9 ;
            P (rejectIdxs, :) = 0;
        end
        XP = X - X*P;
        uuu = ( sum ( P.*P, 2 ) + eps ) .^ 0.5;
        
        % ==== solve Q ====== close form ======
        if beta > 0
            v_inv = 0.5 ./ vvv;
            for n = 1 : nSmp
                vn = v_inv (n);
                hn = XP (:, n);
                Q(:, n) = X.' * ( (XXT + beta*vn.*Ifea) \  hn );
            end
            %             if minX > 0
            %                 Q = max (Q, 0);
            %             end
            vvv = ( sum ( Q.*Q, 1 ) + eps ) .^ 0.5;
            
        else % never update 'Q'
            vvv = 0;
            Q = 0;
        end
        
        % ===== if calculate the objective error ==============
        if ( verb && rem(iter, verb) == 0 )
            rErr = sum (sum( abs( XP - X*Q ) .^ 2 ));
            objErr = rErr + lambda * sum( uuu ) + beta * sum ( vvv );
            fprintf('[%d] : rErr: %f objErr: %f \n', iter, rErr, objErr);
        end
    end
end
end


function [P, Q, objErr] = LF_R21R21_kernel (XTX, maxIter, lambda, beta, verb)
%
% breif:
%  the code (kernel) for Robust Sample Selection via Joint Structured Sparsity.
%
% Notation:
% XTX ... (nSmp x nSmp) is a kernal matrix, nSmp is the number of pixels.
% lambda, beta  ... are two weights parameters.

nSmp = size (XTX, 1);

P = eye (nSmp) * 0.9;
Q = eye (nSmp) * 0.1;
uuu = ones (nSmp, 1);
vvv = ones (nSmp, 1);
Ismp = speye (nSmp, nSmp);

for iter = 1 : maxIter
    % ==== solve P ====== closed form ======
    u_inv = 0.5 ./ uuu;
    U = spdiags (u_inv, 0, nSmp, nSmp);
    P = ( XTX + lambda.*U ) \ ( XTX * (Ismp - Q) );
    uuu = ( sum ( P.*P, 2 ) + eps ) .^ 0.5;
    
    % ==== solve Q ====== closed form ======
    v_inv = 0.5 ./ vvv;
    for n = 1 : nSmp
        vn = v_inv (n);
        pn = P(:, n);
        Q(:, n) = ( XTX + beta*vn.*Ismp ) \ ( XTX(:, n).' - XTX * pn);
    end
    vvv = ( sum ( Q.*Q, 1 ) + eps ) .^ 0.5;
    
    % ===== if calculate the objective error ==============
    if ( verb && rem(iter, verb) == 0 )
        rErr = trace ( (Ismp-P-Q).' * XTX * (Ismp-P-Q) );
        objErr = rErr + lambda * sum(uuu) + beta * sum (vvv);
        fprintf('Kernel : [%d] : rErr: %f\t objErr: %f \n', iter, rErr, objErr);
    end
end
end