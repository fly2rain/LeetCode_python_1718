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
initype = 1;  % choose initialization method
fIter = 0;
weightType = 2;
winSize  = 1;
percent = 0;
lambdaGr = [0.0015,     0.06,     0.05,     0.9,     0.8,   0.3,     0.5];
sigmaGr =  [0.01,       0.1,       0.1,     0.1,     0.1,   0.1,     0.2];
NO_iter = 50;
for noSNR = 1 :  1 % length(SNR)
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
        sigma = sigmaGr(noSNR);
        
         resultName = ['G2-NMF/SNR=' num2str(SNR(noSNR))  '/Initype=' num2str(initype) ...
            ' weightType=' num2str(weightType) ' lambda=' num2str(lambda) ' sigma=' num2str(sigma) ' winSize=' num2str(winSize) ...
            ' percent=' num2str(percent) ' iter=' num2str(iter) '.mat'];
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
        weight=get_GraphWeightMatrix(weightType, V, [nRow,nCol,nBand], [], winSize, sigma, []);
        [W_est, H_est, errObj] = gnmf_sc_norm2(V, W0, H0, weight,...
            'errTol', 1e-7, 'maxIter', 200, 'lambda', lambda, 'alpha', 0, ...
            'normType', 2,'normW_orNot', 1, 'verb', 30, ...
            'myeps', 1e-7, 'firstIterate', fIter);
        H_est(H_est < epss) = 0;
        W_est(W_est < epss) = 0;
        H_est = single(H_est);
        W_est = single(W_est);
        save(resultName, 'H_est', 'W_est');
    end
end
