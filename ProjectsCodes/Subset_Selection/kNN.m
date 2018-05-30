function [acc, accCls] = kNN (XTr, lblTr, XTs, lblTs, k)
% --- Input --------------------------------
% X_tr:   NFea x NSmpTr, the training data set
% lbl_tr: NSmpTr x 1, the labels for training data set
% X_ts:   NFea x NSmpTs, the test data set
% lbl_tr: NSmpTs x 1, the labels for test data set
% k：     the fixed number in a region
% --- Output -------------------------------
% acc:  the accuracy of
% accCls: estimate correct rate for each class
% written by: zfy
% 2012-12-31

% double 2 single, to save memory
XTr = single(XTr);
XTs = single(XTs);

% get parameters
NSmpTs = size(XTs, 2);
lblUqn = unique([lblTr; lblTs]);
NCls = length(lblUqn);
lblTs_hat = zeros (NSmpTs, 1);

NInPat = 10000; % One time process 10000
Npatch = ceil ( NSmpTs / NInPat ); % the number of times to process
NInPatGr = zeros(Npatch, 1);  % the number of samples in one patch (time)

tsPlc = zeros (2, 1); % the beginning and ending places in the sample suqence.
TTT = sum(XTr .* XTr, 1);
for nPat = 1 : Npatch
    if nPat == Npatch
        NInPatGr (nPat) = NSmpTs - (nPat-1) * NInPat;
    else
        NInPatGr (nPat) = NInPat;
    end
    tsPlc(1) = tsPlc(2) + 1;
    tsPlc(2) = tsPlc(1) + NInPatGr(nPat) - 1;    
    smpRange = tsPlc(1) : tsPlc(2);
    
    XTsPat = XTs (:, smpRange);
    % The Euclidean Distance
    MMM = repmat(TTT, [NInPatGr(nPat), 1]) - 2*XTsPat.' * XTr;
    
    lblNum = zeros(NCls, NInPatGr(nPat));
    for ii = 1 : NInPatGr(nPat)
        mm = MMM(ii, :);
        % find the k-th smallest distance
        mmSort = sort(mm, 'ascend');
        m = mmSort(k);
        % find the index for the k smallest distances
        indxTmp = find(mm <= m, k, 'first');
        lblTmp = lblTr(indxTmp); % the labels for the k training points
        for jj = 1 : NCls
            lblNum(jj, ii) = sum(lblUqn(jj)==lblTmp);
        end
    end
    [~, lblTs_hatInPat] = max(lblNum, [], 1);
    lblTs_hat (smpRange) = lblTs_hatInPat;
end

if nargout == 1
    acc = sum(lblUqn(lblTs_hat) == lblTs) / NSmpTs;
    accCls = [];
elseif nargout == 2
    accCls = zeros(NCls, 1);
    for nCls = 1 : NCls
        ind_sub = find (lblTs == nCls);
        NSmpCls = length(ind_sub);
        accCls(nCls) = sum(lblTs_hat(ind_sub)' == lblTs(ind_sub)) / NSmpCls;
    end
    acc = sum(lblTs_hat' == lblTs) / NSmpTs;
end
