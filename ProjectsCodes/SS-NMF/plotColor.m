close all;
% clear all;
addpath('../');
load ('urban.mat');
load ('end4_v9.mat');
lambdaGr = [0.1, 0.12, 0.11, 0.12,  0.11, 0.18, 0.19];
alphaGr  = [0.92, 0.8,  1,    0.86, 0.94, 0.82, 0.88];
% fileNo = 2;
% filename = ['VCA/SNR=30/ iter=1.mat'];
% filename = ['SNR=30 Initype=1 NMF+L1_2 weightType=0 lambda=0 alpha=0.11 winSize=0 sigma=0 percent=0 iter=30.mat'];
% filename = ['SNR=30 Initype=1 NMF+L1 weightType=0 lambda=0 alpha=0.17 winSize=0 sigma=0 percent=0 iter=8.mat'];
filename = ['SNR=30 Initype=1 GNMF+L1 nCenter=8 weightType=7 lambda=0.11 alpha=0.16 winSize=5 sigma=0 percent=30 iter=11.mat'];
load(filename);

epss    = 1e-20;
V       = double(V);
nSmp    = nRow*nCol; % number of the samples

nEnd	= 4; % number of the endnumber
SNR     = inf;

[sub_row, sub_col] = rescale_subplot( nEnd );

% ******* 声明各种向量矩阵, rescale every column of W and W_est to unit length
match = EuError_spectral( W, W_est);
errRmse = zeros(1, nEnd);
errSid = zeros(1, nEnd);
errSad = zeros(1, nEnd);


%% ** 评估和show 光谱估计情况 plot endMember matrix
figure(1)
% yRange = [0.6 0.18 0.65 0.65];
yRange = [1 1 1 1];
for i  = 1 : nEnd
    W_i     =   W(:, match(i,1));
    W_est_i =   W_est(:, match(i,2));    
    % 对齐�?���?    if(rescaleOrNot)
    W_est_i =   W_est_i .* ( max(W_i) / max(W_est_i) );
    % 计算SID差错信息----�?��光谱估计误差的度�?
    errSid( i ) = sid_specInforDiverg( W_i, W_est_i, epss );
    errSad(i)   = sad_specAngDist(W_i, W_est_i, epss);

    
    % show 端元光谱
    subplot( sub_row, sub_col, i);
    plot( W_est_i, 'Linewidth',1 );
    title(cood{i});
    xlabel( 'band number');
    ylabel('reflectance')
    ylim([0 yRange(i)]);
    grid on
    hold on
    plot(1:20:nBand ,W_est_i(1:20:end), 'o', 'Linewidth',1);
    plot( W_i, 'r', 'Linewidth', 1);
end

%% 计算估计的abundant误差，这里采用常用的欧氏误差�?范数误差�?
 % 画groudtruth丰度矩阵�?

 H_tmp = H_est ./ repmat(max(sum(H_est, 1), 1e-20), nEnd, 1);
 for i = 1 : nEnd
     H_i     = H(match(i,1), :);
     H_est_i = H_tmp(match(i,2), :);
     errRmse(i) = ( sum((H_i - H_est_i).^2) / size(H,2) )^0.5;
 end

H_tmp = H_tmp / max(H_tmp(:));
H = H / max(H(:));
A = zeros(nRow, nCol, nEnd);
figure(2)
for i = 1:nEnd  
    subplot( sub_row, sub_col, i);
    H_i = reshape(H(match(i,1), :), [nRow nCol] );
    A(:,:, i) = H_i;
    imshow(H_i, [0 1]);
    title(cood{i});
end
    
A_est = zeros(nRow, nCol, nEnd);
figure(3)
for i = 1:nEnd  
    subplot(sub_row, sub_col, i);
    H_est_i = reshape(H_tmp(match(i,2), :), [nRow nCol] );
    A_est(:,:, i) = H_est_i;
    imshow(H_est_i);
    title(cood{i});
end

showband = [1,2,3];
figure(4)
subplot(1,2, 1);
imshow(A(:,:,showband));
subplot(1,2,2)
imshow(A_est(:,:,showband));
% the first four terms are four endmembers, the last one is the mean 
errSad = [errSad mean(errSad)]
errSid = [errSid mean(errSid)]
errRmse = [errRmse mean(errRmse)]
err = [errSad; errSid; errRmse];
