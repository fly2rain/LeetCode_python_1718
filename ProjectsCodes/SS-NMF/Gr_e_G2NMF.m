% @data: 2012-9-15
close all;  clear all;
addpath('../');
load jasperRidge_2.mat;
load end4_v1.mat;
nSmp    = nRow*nCol; % number of the samples
V = double(V);
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
% choiceOfSNR = 1;
initype = 5;  % choose initialization method
fIter = 0;


%% #1 weightType = 2;
weightType = 2;
NO_iter = 2;
% winSize = 1;
% first lambda， second sigma
lambdaGr = 0.01 : 0.02 : 0.5;
sigmaGr  = 0.01 : 0.02 : 0.2;
atmp = zeros(length(lambdaGr)*length(sigmaGr), nEnd+3);
for i = 1 : length(lambdaGr)
    for j = 1 : length(sigmaGr)
        atmp((i-1)*length(sigmaGr)+j, 1) = lambdaGr(i);
        atmp((i-1)*length(sigmaGr)+j, 2) = sigmaGr(j);
    end
end
cellSad = cell(length(SNR), 2);
for noSNR = 1 : length(SNR)
    cellSad{noSNR, 1} = ['SNR=' num2str(SNR(noSNR))];
    cellSad{noSNR, 2} =  atmp;
end
cellRmse = cell(length(SNR), 2);
for noSNR = 1 : length(SNR)
    cellRmse{noSNR, 1} = ['SNR=' num2str(SNR(noSNR))];
    cellRmse{noSNR, 2} =  atmp;
end
errSad = zeros(NO_iter, nEnd + 1);
errRmse = zeros(NO_iter, nEnd + 1);

for noSNR = 1 : length(SNR)
    for noLambda = 1 : length(lambdaGr)
        for noSigma = 1 : length(sigmaGr)
            for iter = 1 : NO_iter
                lambda = lambdaGr(noLambda);
                sigma = sigmaGr(noSigma);              
                resultName = ['SNR=' num2str(SNR(noSNR)) ' initype=' num2str(initype)  ' GNMF weightType='...
                    num2str(weightType)  ' lambda= ' num2str(lambda) ' sigma=' num2str(sigma) ...
                    ' iter=' num2str(iter)  '.mat'];
                disp(resultName);
                load(resultName);
                
                resultName = ['../G2NMF/GNMF SNR=' num2str(SNR(noSNR)) ' initype=' num2str(initype)  ' weightType='...
                    num2str(weightType)  ' lambda= ' num2str(lambda) ' sigma=' num2str(sigma) ...
                    ' iter=' num2str(iter)  '.mat'];
                save(resultName, 'H_est', 'W_est');
                
%                 match = EuError_spectral( W, W_est);
%                 for i = 1 : nEnd
%                     W_i     =   W(:, match(i,1));
%                     W_est_i =   W_est(:, match(i,2));
%                     errSad(iter, i)   = sad_specAngDist(W_i, W_est_i, epss);
%                 end
%                 
%                 H_tmp = H_est ./ repmat(max(sum(H_est, 1), 1e-20), nEnd, 1);
%                 for i = 1 : nEnd
%                     H_i     = H(match(i,1), :);
%                     H_est_i = H_tmp(match(i,2), :);
%                     errRmse(iter, i) = ( sum((H_i - H_est_i).^2) / size(H,2) )^0.5;
%                 end                
            end
%             errSad(:, nEnd+1) = mean(errSad(:, 1:nEnd), 2);
%             errRmse(:, nEnd+1) = mean(errRmse(:, 1:nEnd), 2);
%             
%             cellSad{noSNR, 2}((noLambda-1)*length(sigmaGr), 2:end) = mean(errSad, 1);
%             cellRmse{noSNR, 2}((noLambda-1)*length(sigmaGr), 2:end) = mean(errRmse, 1);
        end
    end
end
save('res_GNMF_gauss.mat', 'cellSad', 'cellRmse');


