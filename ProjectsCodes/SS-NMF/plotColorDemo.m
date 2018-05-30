close all;
% clear all;
addpath('../');
load urban.mat;
load end4_v9.mat;
epss    = 1e-20;
V       = double(V);
nSmp    = nRow*nCol; % number of the samples
nEnd	= 4; % number of the endnumber
H3d =  reshape(H', [nRow nCol nEnd]);

% figure(1)
% orgImg = imread('samson_1.bmp');
% show original Pseudo-color image of Hyperspectral data
% imshow(orgImg);

segH = zeros(size(H));
maxAbund = max(H,[],1);
for i = 1 : nEnd
    a = H(i,:);
    a(a ~= maxAbund) = 0;
    a(a == maxAbund & a ~= 0) = 1;
    segH(i, :) =  a;
end
segH = reshape(segH', [nRow, nCol, nEnd]);

% figure
% imshow(segH, []);
imwrite(segH(:,:,1:3), 'segmentaion.bmp', 'bmp');

% figure 
% imshow(reshape(H', [nRow nCol nEnd]));
imwrite(H3d(:,:,1:3), 'unmixingColor.bmp', 'bmp');
imwrite(H3d(:,:,1), 'unmixing_1.bmp', 'bmp');
imwrite(H3d(:,:,2), 'unmixing_2.bmp', 'bmp');
imwrite(H3d(:,:,3), 'unmixing_3.bmp', 'bmp');
% reg1_orgImg = orgImg(1:45, 17:17+45, :);
% reg1_H3d    = H3d(1:45, 17:17+45, :);
% reg1_segH   = segH(1:45, 17:17+45, :);
% subplot(131)
% imshow(reg1_orgImg)
% subplot(132)
% imshow(reg1_H3d )
% subplot(133)
% imshow(reg1_segH)




