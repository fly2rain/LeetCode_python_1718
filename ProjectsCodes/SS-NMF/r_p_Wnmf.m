% 2013-07-17, by fyzhu, email: fyzhu@nlpr.ia.ac.cn.
% choosing the value of Lagrange multiplier parameters for WNMF,
% ref to paper: < Enhancing Spectral Unmixing by Local Neighborhood Weights>
% run
close all;  clear all;
addpath('../../', '../');
load    urban.mat;
nSmp    = nRow*nCol; % number of the samples
V       = double(V);
epss    = 1e-20;
nEnd    = 4;
% =========== important parameters ==========
SNRGr = [ inf, ...        % #1
    30, ...         % #2
    25, ...         % #3
    20, ...         % #4
    15, ...         % #5
    10, ...         % #6
    8, ...          % #7
    ];

initype     = 1;  % choose initialization method
fIter       = 0;
weightType  = 9;
winSize     = 1;
percent     = 0;
% SNR       =   inf,    30,     25,     20,     15,     10,     8
lambdaGr    = [0.0008,  0.0015, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001];

NO_iter = 20;
for noSNR = 1 : length(SNRGr)    
    % ===================== add noise ===================
    if SNRGr(noSNR) == inf
        V = max(V,0);
    else
        V = awgn(V, SNRGr(noSNR), 'measured', 'linear');
        V = max(V,0);
    end
    
    % Concstruct the graph
    weight = get_GraphWeightMatrix(weightType, V, [nRow,nCol,nBand], [], winSize, 0, []);
    
    lambda  = lambdaGr(noSNR);
    
    for iter = 1 : NO_iter        
        resultName = ['WNMF/SNR=' num2str(SNRGr(noSNR))  ' Initype=' num2str(initype) ...
            ' lambda=' num2str(lambda) ' winSize=' num2str(winSize) ...
            ' iter=' num2str(iter) '.mat'];
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
        
        % iteration --------
        [W_est, H_est, errObj] = ss_nmf_v1 (V, W0, H0, weight,...
            'errTol', 1e-7, 'maxIter', 200, 'lambda', lambda, 'alpha', 0, ...
            'normType', 2,'normW_orNot', 1, 'verb', 20, ...
            'myeps', 1e-7, 'firstIterate', 0);
        
        H_est(H_est < epss) = 0;
        W_est(W_est < epss) = 0;
        H_est = single(H_est);
        W_est = single(W_est);
        save(resultName, 'H_est', 'W_est');        
    end
end