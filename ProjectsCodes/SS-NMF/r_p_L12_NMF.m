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
initype = 1;  % choose initialization method
fIter = 0;
% alphaGr = 0.1:0.02:1.1;
alphaGr = [0.39, 0.37, 0.45, 0.35, 0.55, 0.59, 0.53];
NO_iter = 50;
for noSNR = 1:2 % 2 : length(SNR)
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
        alpha = alphaGr(noSNR);        
       resultName = ['L12-NMF/SNR=' num2str(SNR(noSNR)) '/initype=' num2str(initype)  ...
             ' alpha=' num2str(alpha) ' iter=' num2str(iter)  '.mat'];
        disp(resultName);
        
        % ===================================================
        % **************  initialization  W and H ***********
%         rowSub = [19    38      8      19];
%         colSub = [5     35      57     75];
        rowSub = [];
        colSub = [];
        selectIndx = sub2ind([nRow nCol], rowSub, colSub);
        [W0, H0] = allInitMethods(initype, V, nEnd, selectIndx, epss);
        [W_est, H_est, errObj] = gnmf_L1in2_norm2(V, W0, H0, [],...
                'errTol', 1e-6, 'maxIter', 200, 'lambda', 0, 'alpha', alpha, ...
                'normType', 2,'normW_orNot', 1, 'verb', 10, ...
                'myeps', 1e-10);
        H_est(H_est < epss) = 0;
        W_est(W_est < epss) = 0;
        H_est = single(H_est);
        W_est = single(W_est);
        save(resultName, 'H_est', 'W_est');
    end
end
