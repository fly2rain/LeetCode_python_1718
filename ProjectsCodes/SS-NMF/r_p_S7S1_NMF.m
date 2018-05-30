% @data: 2012-9-15
close all;  clear all;
addpath('../');
epss    = 1e-20;
nEnd    = 4;
% =========== important parameters ==========
SNR = [ inf, ...        % #1 
        30, ...         % #2 
        25, ...         % #3
        20, ...         % #4
        15, ...         % #5
        10, ...         % #6
        8, ...          % #7        
      ];
% noSNR = 1;
weightType  = 7;
winSize     = 5;
percent     = 30;
initype     = 1;
fIter       = 0;
lambdaGr = [0.15,   0.11,   0.11,   0.09,   0.095,  0.13,   0.13];
alphaGr  = [0.16,   0.16,   0.16,   0.16,   0.165,  0.18,   0.18];
nCenterGr= [8   8   8   8   8   4   4];
NO_iter = 50;
for noSNR = 1:length(SNR)
    load urban.mat;
    nSmp    = nRow*nCol; % number of the samples
    V = double(V);
    % ===================== add noise ===================
    if SNR(noSNR) == inf
        V = max(V,0);
    else
        V = awgn(V, SNR(noSNR), 'measured', 'linear');
        V = max(V,0);
    end   
   
    for iter = 1 : NO_iter
        lambda = lambdaGr(noSNR);
        alpha = alphaGr(noSNR);
        nCenter = nCenterGr(noSNR);   
        load(['kmeans/1 SNR=' num2str(SNR(noSNR)) ' kmeans_cos nCenter=' num2str(nCenter) '.mat']);
        resultName = ['G6NMF-L1/SNR=' num2str(SNR(noSNR)) '/initype=' num2str(initype)  ' nCenter=' num2str(nCenter) ' weightType='...
                    num2str(weightType)  ' lambda= ' num2str(lambda) ' alpha=' num2str(alpha) ...
                    ' winSize=' num2str(winSize) ' percent=' num2str(percent) ' iter=' num2str(iter)  '.mat'];
        disp('==========================');
        disp(resultName);
        
        % ===================================================
        % **************  initialization  W and H ***********
%         rowSub = [19    38      8      19];
%         colSub = [5     35      57     75];
        rowSub = [];
        colSub = [];
        selectIndx = sub2ind([nRow nCol], rowSub, colSub);
        [W0, H0] = allInitMethods(initype, V, nEnd, selectIndx, epss);
        weight=get_GraphWeightMatrix(weightType, V, [nRow,nCol,nBand], IDX, winSize, 0, percent);
        [W_est, H_est, errObj] = gnmf_sc_norm2(V, W0, H0, weight,...
            'errTol', 1e-7, 'maxIter', 200, 'lambda', lambda, 'alpha', alpha, ...
            'normType', 2,'normW_orNot', 1, 'verb', 30, ...
            'myeps', 1e-7, 'firstIterate', fIter);
        H_est(H_est < epss) = 0;
        W_est(W_est < epss) = 0;
        H_est = single(H_est);
        W_est = single(W_est);
        
        save(resultName, 'H_est', 'W_est');
    end
end
