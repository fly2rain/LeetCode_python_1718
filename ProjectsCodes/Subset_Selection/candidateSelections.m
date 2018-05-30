function [candIdxs, candVals] = candidateSelections (A, Nslcts, Ratio, isRowSparse)
%
% Breif:
%      Select the candidate Idxs and Vals 
%
% Notation: (take sample selection for example)
% A ... (nSmp x nSmp) coefficient matrix with few nonzero rows and computes 
%           the indices of the nonzero rows.
% Nslcts ... (1 x 1) the number of selected samples
%           the most representive and informative features.
% Ratio ... (1 x 1) the ratio that is more than the selected samples
%
% isRowSparse ... (1 x 1) A is row sparse or column sparse.
%
%  version 1.0 -- 2014-7-26
%
%  Written by Feiyun Zhu

if (nargin < 4)
    isRowSparse = 1;
end

if (nargin < 3)
    Ratio = 1;
end

NSmp = size(A, 1);
CandNum = Nslcts * (1 + Ratio);

if CandNum >= NSmp
    warning ('Overflow: too many are selected.\n');
end

if isRowSparse % row sparse
%     a = sum (A.*A, 2);
    a = sum (abs(A), 2);
    [vals, idxs] = sort (a, 'descend');
    candIdxs = idxs (1:CandNum);
    candVals = vals (1:CandNum);
    
else % ===== column sparse
%     a = sum (A.*A, 1);
    a = sum (abs(A), 1);
    [vals, idxs] = sort (a, 'descend');
    candIdxs = idxs (1:CandNum);
    candVals = vals (1:CandNum);
end
