function FotN = F_fot (OtN, md, RBFbnd)
% Breif:
%   Construt the basic features for the critic (value) function based on
%   the input state vectors simultaneously for all the N samples.
%
% Input parameters ################################################
% OtN:   (Lo x N) the state feature vectors for N samples
% md:    structure, including important arguments related to vector dimensions
%
% Output parameters ################################################
% FotN: (Lgo x N) the construct basic feature vectors for the value function.
%
% References:
%     [1]
%
%  version 1.0 - 12/09/2015
%
%  Written by Feiyun Zhu (fyzhu0915@gmail.com)

N = size (OtN, 2);
if nargin < 3
    RBFbnd = 1;
end

% Choices: 1) itself, 2) normal, 3) polynomial, 4) RBF, 5) sparse
switch md.FoChoice
    case 'itself'     % itself
        FotN = OtN;
    case 'regular'    % [Ot, Ot^2, product(combine(Ot,2))]
        if md.Lo == 1
            FotN = vertcat ( OtN, OtN.^2 );
        elseif md.Lo == 3
            FotN = vertcat ( OtN, OtN.^2, ...
                OtN(1,:).*OtN(2,:), OtN(1,:).*OtN(3,:), OtN(2,:).*OtN(3,:) );
        end        
    case 'polynomial' % the polynomial feature
        OtN = OtN / max (max(abs(OtN(:)), 1E-10));
        FotN = F_fea4ValueFunc_polynomial (OtN, md.order);
    case 'RBF'
        FotN = F_fea4ValueFunc_RBF (OtN, md.order, RBFbnd);
    case 'sparse'     % the very sparse feature
        Lo   = md.Lo;
        OtN  = fix(OtN ./ 0.2);
        FotN = zeros (md.Lfo, N);
        idxN = OtN(1,:);
        for ii = 1 : (Lo-1)
            idxN = idxN + OtN(ii+1,:)*(Lo^ii);
        end
        FotN(idxN) = 1;
    otherwise
        error ('Wrong choice of Lfo.\n');
end

% figure,
% subplot (121)
% hist (OtN(:), 100);
% title ('hist of states Ot')
% subplot (122)
% hist (FotN(:), 100)
% title ('hist of polynomial feature of Ot')
