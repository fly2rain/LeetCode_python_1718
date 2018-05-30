% 2013-07-17, by fyzhu, email: fyzhu@nlpr.ia.ac.cn.
% choosing the value of Lagrange multiplier parameters for EDC_nmf,
% ref to paper: < An Endmember Dissimilarity Constrained Non-Negative Matrix
% Factorization Method for Hyperspectral Unmixing>
% try run

close all;  clear all;
addpath('../../', '../');
epss    = 1e-20;
nEnd    = 4;

% =========== important parameters ==========
SNRGr = [ inf, ...  % #1
    30, ...         % #2
    25, ...         % #3
    20, ...         % #4
    15, ...         % #5
    10, ...         % #6
    8, ...          % #7
    ];

initype     = 1;  % choose initialization method
fIter       = 0;

% SNR       =   inf,    30,     25,     20,     15,     10,     8
lambdaGr    = [4500,   4000,   4500,   3000,   3000,   4500,  1000];

NO_iter = 20;
for noSNR = 5 % : length(SNRGr)
    load    urban.mat;
    nSmp    = nRow*nCol; % number of the samples
    V       = double(V);
    % ===================== add noise ===================
    if SNRGr(noSNR) == inf
        V = max(V,0);
    else
        V = awgn(V, SNRGr(noSNR), 'measured', 'linear');
        V = max(V,0);
    end
    
    lambda  = lambdaGr(noSNR);
    
    for iter = 1 : NO_iter
        resultName = ['EDC_nmf/SNR=' num2str(SNRGr(noSNR))  ' Initype=' num2str(initype) ...
            ' lambda=' num2str(lambda) ' iter=' num2str(iter) '.mat'];
        disp('==========================');
        disp(resultName);
        
        % ===================================================
        % **************  initialization  W and H ***********
        rowSub = [19    38      8      19];
        colSub = [5     35      57     75];
        %             rowSub = [];
        %             colSub = [];
        selectIndx = sub2ind([nRow nCol], rowSub, colSub);
        [W0, H0] = allInitMethods(initype, V, nEnd, selectIndx, epss);
        
        %
        [W_est, H_est, errObj] = EDC_nmf (V, W0, H0,...
            'errTol', 1e-4, 'maxIter', 200, 'lambda', lambda, 'normType', 2, ...
            'normW', 1, 'verb', 10, 'myeps', 1e-10, 'firstIterate', 0);
        
        H_est(H_est < epss) = 0;
        W_est(W_est < epss) = 0;
        H_est = single(H_est);
        W_est = single(W_est);
        save(resultName, 'H_est', 'W_est');
    end
end
