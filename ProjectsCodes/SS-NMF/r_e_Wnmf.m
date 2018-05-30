% 2013-07-17, by fyzhu, email: fyzhu@nlpr.ia.ac.cn.
% choosing the value of Lagrange multiplier parameters for WNMF,
% ref to paper: < Enhancing Spectral Unmixing by Local Neighborhood Weights>
% test

close all;  clear all;
addpath('../../', '../');
load urban.mat;
load end4_v9.mat;

V       = double(V);
nSmp    = nRow*nCol; % number of the samples
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

initype     = 1;  % choose initialization method
fIter       = 0;
weightType  = 9;
winSize     = 1;
percent     = 0;

% SNR       =   inf,    30,     25,     20,     15,     10,     8
lambdaGr    = [0.0008,  0.0015, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001];

NO_iter = 20;

resObjMean  = zeros(length(SNR), 2*(nEnd+1)+1);  % sad rmse 
resObjStd   = zeros(length(SNR), 2*(nEnd+1)+1);  % sad rmse 

errSad = zeros(NO_iter, nEnd + 1);
errRmse = zeros(NO_iter, nEnd + 1);
for noSNR = 1 : length(SNR)
    lambda = lambdaGr(noSNR);
    
    for iter = 1 : NO_iter
        resultName = ['WNMF/SNR=' num2str(SNR(noSNR))  ' Initype=' num2str(initype) ...
            ' lambda=' num2str(lambda) ' winSize=' num2str(winSize) ...
            ' iter=' num2str(iter) '.mat'];
        
        disp(resultName);
        load(resultName);
        
        W_est = double (W_est);
        H_est = double (H_est);
        
        match = EuError_spectral( W, W_est);
        for i = 1 : nEnd
            W_i     =   W(:, match(i,1));
            W_est_i =   W_est(:, match(i,2));
            errSad(iter, i)  = sad_specAngDist(W_i, W_est_i, epss);
        end
        
        H_tmp = H_est ./ repmat(max(sum(H_est, 1), 1e-20), nEnd, 1);
        for i = 1 : nEnd
            H_i     = H(match(i,1), :);
            H_est_i = H_tmp(match(i,2), :);
            errRmse(iter, i) = ( sum((H_i - H_est_i).^2) / size(H,2) )^0.5;
        end
    end
    
    errSad(:, nEnd+1) = mean(errSad(:, 1:nEnd), 2);
    errRmse(:, nEnd+1) = mean(errRmse(:, 1:nEnd), 2);
    
    
    resObjMean(noSNR, 1:nEnd+1)     = mean(errSad, 1);
    resObjMean(noSNR, nEnd+3:end)   = mean(errRmse, 1);
    
    resObjStd(noSNR, 1:nEnd+1)      = std(errSad, 1, 1);
    resObjStd(noSNR, nEnd+3:end)    = std(errRmse, 1, 1);
end


%% build tongji data for plot
noAlg    =  7;
storeTongji;

