function [Xtr, Ytr, Xts, Yts] = chooseTrainTestSets (X, Y, nChsPerClss)
% X: L x N, each  column is a sample vector,
% Y: N, the corresponding labels
% nChsPerClss: the number of selected samples for each class.

lblSet = unique (Y);
[NFea, NSmp] = size (X);

NClass = length (lblSet);

% select the same number of samples for each class
if length (nChsPerClss) == 1 
    NTr = nChsPerClss * NClass;
    nChsPerClss = repmat (nChsPerClss, [NClass, 1]);
else % select the same number of samples for each class
    NTr = sum (nChsPerClss);
end
NTs = NSmp - NTr;

Xtr = zeros (NFea, NTr);
Ytr = zeros (NTr, 1);
trPlc = zeros (2, 1);

Xts = zeros (NFea, NTs);
Yts = zeros (NTs, 1);
tsPlc = zeros (2, 1);

for nn = 1 : NClass
    trPlc(1) = trPlc(2) + 1;
    tsPlc(1) = tsPlc(2) + 1;
    
    nClsIdx = lblSet (nn);
    cSet = find (Y == nClsIdx);
    
    nSmpCls = length (cSet);
    
    trPlc(2) = trPlc(1) + nChsPerClss(nn) - 1;
    tsPlc(2) = tsPlc(1) + nSmpCls - nChsPerClss(nn) - 1;
    
    trIdx = randperm (nSmpCls, nChsPerClss(nn));
    tsIdx = setdiff (1:nSmpCls, trIdx);
    
    trIdx = cSet (trIdx);
    tsIdx = cSet (tsIdx);
    
    Xtr (:, trPlc(1):trPlc(2)) = X (:, trIdx);
    Ytr (trPlc(1):trPlc(2)) = nClsIdx;
    %     Ytr (trPlc(1):trPlc(2)) = Y (trIdx);
    
    Xts (:, tsPlc(1):tsPlc(2)) = X (:, tsIdx);
    Yts (tsPlc(1):tsPlc(2)) = nClsIdx;
    %     Yts (tsPlc(1):tsPlc(2)) = Y (tsIdx);
end