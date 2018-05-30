addpath('../../');
% NMF, l1-nmf, l1/2-nmf, EDC_NMF, DgS_nmf, reference
figPathes = {
    '../../../SS-NMF_for_HU/urban_end4_iter50/NMF/SNR=Inf/initype=1 normType=2 iter=1.mat' , ...
    '../../../SS-NMF_for_HU/urban_end4_iter50/L1-NMF/SNR=Inf/initype=1 alpha=0.15 iter=1.mat' , ...
    '../iter20/L05_nmf/SNR=Inf Initype=1 lambda=0.03 maxIter=300 iter=19.mat' , ... %    
    '../../../SS-NMF_for_HU/urban_end4_iter50/EDC_nmf/SNR=Inf Initype=1 lambda=4500 iter=2.mat' , ...
    '../iter20/DgS_nmf/SNR=Inf iniType=1 lambda=0.076 ratIter=0.48 maxIter=600 iter=3.mat' ...
    };


load ../end4_v9.mat;
A = double (H);
M = double (W);
load ../urban_noiseFree.mat;

gapRow = 9; gapCol = 9;   % 图与图之间的间隔
nEnd = 4;   nAlg = 5;     % endmember个数�?算法的个�?figRow = nEnd; figCol = 6;   % 总共几行，几列图

A3D = reshape(A', [nRow nCol nEnd]);
endShunXu = [1 2 3 4];
endRgb = [1 2 3];
% endRgb = [3 1 2];

% create the large matrix for bw and rgb
A_rgb = ones(nRow, figCol*(nCol+gapCol)-gapCol, 3);
A_chaRgb = ones(2*nRow+gapRow, figCol*(nCol+gapCol)-gapCol, 3);

A_bw = ones(figRow*(nRow+gapRow)-gapRow, figCol*(nCol+gapCol)-gapCol);
A_chaBw = ones(nRow, (figCol-1)*(nCol+gapCol)-gapCol);

% put the ground truth in for H_color in the last subfigure
A_rgb(:, (figCol-1)*(nCol+gapCol)+(1:nCol), :) = A3D(:, :, endRgb);
A_chaRgb(158:464, (figCol-1)*(nCol+gapCol)+(1:nCol), :) = A3D(:, :, endRgb);
j = figCol;
for i = 1 : nEnd
    A_bw((i-1)*(nRow+gapRow)+(1:nRow), ...
        (j-1)*(nCol+gapCol)+(1:nCol), :) = A3D(:, :, endShunXu(i));
end


for j = 1 : nAlg
    load (figPathes{j});
    if j == 1 % || j == 2 || j == 4
        A_est = double (H_est);
        M_est = double (W_est);
    end
        
    A_est = A_est ./ repmat(sum(A_est), [nEnd, 1]);
    match = EuError_spectral(M, M_est);
    
    A3D_est = reshape(A_est', [nRow nCol nEnd]);
    A_rgb(:, (j-1)*(nCol+gapCol)+(1:nCol), :) = A3D_est(:, :, match(endRgb, 2));
    
    A_chaRgb(1:nRow, (j-1)*(nCol+gapCol)+(1:nCol), :) = A3D_est(:, :, match(endRgb, 2));
    A_chaRgb(nRow+gapRow+(1:nRow), (j-1)*(nCol+gapCol)+(1:nCol), :) = 1.2*abs(A3D(:,:,endRgb) - A3D_est(:, :, match(endRgb, 2)));
    
    chaTmp = double(A - A_est(match(:, 2), :));
    chaTmp = sum(chaTmp.^2, 1).^0.5;    
    A_chaBw(:, (j-1)*(nCol+gapCol)+(1:nCol)) = reshape (chaTmp, [nRow, nCol]);
    for i = 1 : nEnd
        ix = endShunXu(i);
        A_bw((ix-1)*(nRow+gapRow)+(1:nRow), ...
            (j-1)*(nCol+gapCol)+(1:nCol), :) = A3D_est(:, :, match(i, 2));
    end
end

A_chaBw = A_chaBw ./ max (A_chaBw(:));
A_rgb = single (A_rgb);
A_bw  = single (A_bw);

% imwrite(A_chaRgb, ['urban_abundRgbCha.jpg'], 'jpg');
imwrite(A_chaRgb, ['urban_abundRgb&Cha.bmp'], 'bmp');
imwrite(A_bw, ['urban_abundBw.bmp'], 'bmp');

figure(2),
% subplot (311),
% imshow(A_rgb);
% subplot (312),
imshow(A_chaRgb);
% subplot (313),
% imshow(A_chaBw);

figure(3), imshow(A_bw);
