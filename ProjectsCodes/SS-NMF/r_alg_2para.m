% @data: 2013-3-9 this program is to compare performance vs. sparse
close all;  clear all;
addpath('../');
epss    = 1e-20;
nEnd    = 4;
% =========== important parameters ==========
weightType  = 7;
nCenter     = 8;
winSize     = 5;
percent     = 30;
initype     = 5;
fIter       = 0;
sigma       = 0.01;
lambda = 0.15;
alpha = 0.16;
paraProb = [-2:0.5:2];
grphChoice = 2.^paraProb .* lambda;
sprsChoice = 2.^paraProb .* alpha;
NO_iter = 1;
maxIter = 200;

% load data set
load urban.mat;
nSmp    = nRow*nCol; % number of the samples
V = double(V);
V = max(V,0);
load(['1 SNR=' num2str(inf) ' kmeans_corr'  ' nCenter=' num2str(nCenter) '.mat']);
weight_ss=get_GraphWeightMatrix(weightType, V, [nRow,nCol,nBand], IDX, winSize, 0, percent);
weight_g=get_GraphWeightMatrix(2, V, [nRow,nCol,nBand], [], winSize, sigma, []);
% **************  initialization  W0 and H0 *********** 
for noPara = 1 : length(sprsChoice) % 1:length(SNR)    
    %% 2-nd SS-NMF ===================    
    lambda = grphChoice(noPara);
    alpha = sprsChoice(noPara);
    resultName = ['Perform2Paras/L1-G6NMF lambda=' num2str(lambda) ...
        ' alpha=' num2str(alpha) ' noPara=' num2str(noPara) '.mat'];
    disp('==========================');
    disp(resultName);
    
    [W0, H0] = allInitMethods(1, V, nEnd, [], epss);  
    [W_est, H_est, errObj] = gnmf_sc_norm2(V, W0, H0, weight_ss,...
        'errTol', 1e-7, 'maxIter', maxIter, 'lambda', lambda, 'alpha', alpha, ...
        'normType', 2,'normW_orNot', 1, 'verb', 30, ...
        'myeps', 1e-7, 'firstIterate', fIter);
    H_est(H_est < epss) = 0;
    W_est(W_est < epss) = 0;
    H_est = single(H_est);
    W_est = single(W_est);
    
    save(resultName, 'H_est', 'W_est');

    %% 4-th L1-NMF
    alpha = sprsChoice(noPara);
    resultName = ['Perform2Paras/L1-NMF alpha=' num2str(alpha) '.mat'];
    disp('==========================');
    disp(resultName);
    
    [W0, H0] = allInitMethods(1, V, nEnd, [], epss);
    [W_est, H_est, errObj] = gnmf_sc_norm2(V, W0, H0, [],...
        'errTol', 1e-7, 'maxIter', maxIter, 'lambda', 0, 'alpha', alpha, ...
        'normType', 2,'normW_orNot', 1, 'verb', 30, ...
        'myeps', 1e-7, 'firstIterate', fIter);
    H_est(H_est < epss) = 0;
    W_est(W_est < epss) = 0;
    H_est = single(H_est);
    W_est = single(W_est);
    save(resultName, 'H_est', 'W_est');
    
    %% 5-th L12-NMF    
    alpha = sprsChoice(noPara);
    resultName = ['Perform2Paras/L12-NMF alpha=' num2str(alpha)  '.mat'];
    disp('==========================');
    disp(resultName);
    
    [W0, H0] = allInitMethods(1, V, nEnd, [], epss);
    [W_est, H_est, errObj] = gnmf_L1in2_norm2(V, W0, H0, [],...
        'errTol', 1e-6, 'maxIter', maxIter, 'lambda', 0, 'alpha', alpha, ...
        'normType', 2,'normW_orNot', 1, 'verb', 30, ...
        'myeps', 1e-10);
    H_est(H_est < epss) = 0;
    W_est(W_est < epss) = 0;
    H_est = single(H_est);
    W_est = single(W_est);
    save(resultName, 'H_est', 'W_est');
     
    
    %% 6-th G-NMF
    lambda = grphChoice(noPara);
    resultName = ['Perform2Paras/G-NMF lambda=' num2str(lambda) '.mat'];
    disp('==========================');
    disp(resultName);
    
    [W0, H0] = allInitMethods(1, V, nEnd, [], epss);
    [W_est, H_est, errObj] = gnmf_sc_norm2(V, W0, H0, weight_g,...
        'errTol', 1e-7, 'maxIter', maxIter, 'lambda', lambda, 'alpha', 0, ...
        'normType', 2,'normW_orNot', 1, 'verb', 30, ...
        'myeps', 1e-7, 'firstIterate', fIter);
    H_est(H_est < epss) = 0;
    W_est(W_est < epss) = 0;
    H_est = single(H_est);
    W_est = single(W_est);
    save(resultName, 'H_est', 'W_est'); 
end
