function r = rmse_abunEucMetric(H, H_hat)
%  r = rmse_abunEucMetric(H, H_hat)
% parameters **************************************************************
    %  H          is the standard Abundant matrix, or a abudant row
    %  H_hat      is the estimated Abundant matrix, or a abundant row
% *************************************************************************
    %  author : zhu feiyun
    %  time   : 2012-2-22
    %  version: 1.0
    
[ end_num pix_num ] = size(H);
r = zeros( 1, end_num );
diver = zeros( end_num );
for i = 1 : end_num
    for j = 1 : end_num
        diver(i, j) = norm( H(i,:) - H_hat(j,:) ) / sqrt( pix_num );
    end
end
r = min(diver);
