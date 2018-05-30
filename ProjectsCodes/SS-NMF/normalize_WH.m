function [W, H] = normalize_WH(W, H, normType, normW_orNot)
% [W, H] = normalize_WH(W, H, normType, normW_orNot)
%
% Normalize rows of H using type which can be:
%  1   - use 1-norm [default]
%  2   - use 2-norm
% which is lighted with Caideng's code
% -----------------------------
% Version:      1
% Modified by: Zhu Feiyun
% Date:        2012-3-21

if nargin == 2
    normType = 2;
    normW_orNot = 1;
end
if nargin == 3
    normW_orNot = 1;
end

nEnd = size(H,1);
if(nEnd ~= size(W,2))
    error('row_of_H == column_of_W not satisfied\n');
end

switch normType
    case 1 % l1 norm rescale
        if normW_orNot % resacle W
            norms = max(1e-15, sum(abs(W), 1))';
            W   = W * spdiags(norms.^-1, 0, nEnd, nEnd);
            H   = spdiags(norms, 0, nEnd, nEnd) * H;
        else % rescale H
            norms = max(1e-15, sum(abs(H), 2));
            W   = W * spdiags(norms, 0, nEnd, nEnd);
            H   = spdiags(norms.^-1, 0, nEnd, nEnd) * H;
        end
    case 2 % l2 norm rescale
        if normW_orNot % rescale W
            norms = max(1e-15, sqrt(sum(W.^2, 1)))';
            W   = W * spdiags(norms.^-1, 0, nEnd, nEnd);
            H   = spdiags(norms, 0, nEnd, nEnd) * H;
        else % rescale H
            norms = max(1e-15, sqrt(sum(H.^2, 2)));
            W   = W * spdiags(norms, 0, nEnd, nEnd);
            H   = spdiags(norms.^-1, 0, nEnd, nEnd) * H;
        end
    otherwise
        error('normType must be 1 or 2');
end
W = full(W);
H = full(H);
