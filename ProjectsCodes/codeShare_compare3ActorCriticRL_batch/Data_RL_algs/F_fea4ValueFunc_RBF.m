function Y = F_fea4ValueFunc_RBF (X, order, Bnd, sigma2 )
% Breif:
%   Construct the n-order Radial Basis Function (RBF) feature simultaneously
%   for N input samples. The elements in the centers are evenly distributed
%   in the range of [-Bnd, Bnd].
%
% Input parameters ################################################
% X     ... (L x N) contains N samples; each sample has L feature elements.
% order ... (1 x 1) the order value of the RBF basis
% Bnd   ... (1 x 1) all the evenly distributed centers are in the range of [-Bnd, Bnd]
% sigma2... (1 x 1) the windows width of the RBF basic function
%
% Output parameters ################################################
% Y     ... (Lf x N) the constructed feature vector
%
% References:
%     [1] George Konidaris. "Value Function Approximation in Reinforcement
%          Learning using the Fourier Basis", AAAI, 2011.
%
% version 1.0 -- 02/07/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

% set the defult input argumnets ------------
if nargin < 4
    if nargin < 3
        if nargin < 2
            order = 3;
        end
        Bnd = 1;
    end
    sigma2 = 2 / (order - 1);
end

[Lo, N] = size (X); % X contains N samples; each sample has L feature elements. 
Lfo     = order ^ Lo; % the length of  constructed RBF features. 
% get the evenly distributed centers in the state space
xCenters = F_getEvenlyDistributedCenters (X, order, Bnd);

% the constructed RBF features
Y = zeros (Lfo, N);
for n = 1 : N
    xn = X (:, n); % thr n-th sample 
    
    yn = sum( ( repmat(xn,1,Lfo) - xCenters ).^2, 1 ).';
    yn = -yn ./ (2*sigma2);
    Y(:,n) = 1/sqrt(2*pi*sigma2) * exp( yn );
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xCenters = F_getEvenlyDistributedCenters (X, order, Bnd)
% Breif:
%   Get the evenly distributed centers, with each center elements in [-Bnd,Bnd]
% Note that: 
%   Please check the previouse function for the detailed description of the
%   input arguments. 
% 
% version 1.0 -- 02/07/2016
%
% Written by Feiyun Zhu (fyzhu0915@gmail.com)

Lo  = size (X, 1); % X contains N samples; each sample has L feature elements.
Lfo = order ^ Lo; % the length of the constructed RDF feature
C   = zeros (Lo, Lfo); % the order matrix to construct the evenly distributed centers
idxSequence = 0 : Lfo-1;  % the sequence to construct the order matrix C effectively.

% get the evenly distributed indexes in the state spaces
denom = order ^ (Lo-1);
for i = 1 : Lo-1
    C(Lo-i+1,:) = fix ( idxSequence ./ denom );
    idxSequence = rem ( idxSequence, denom );
    denom       = denom / order;
end
C(1,:) = idxSequence;

% the interval between the adjacent centers, which are evenly distributed.
xDelta = 2 * Bnd / (order - 1);
% all the evenly distributed centers
xCenters = -Bnd + xDelta*C;
end