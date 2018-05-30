function W = get_GraphWeightMatrix(weightType, data, dims, IDX, winSize, sigma, percent)
% Author: Zhu Feiyun
% Date:   2012-7-14
% ----------------------------------------------------------------------
% ========================== weigth matrix =============================
nSmp = size(data, 2);

switch weightType
    
    %% caideng 2011 pami
    case 1 % 0-1 weighting:
        [ri, ci, val] = mexWeightCD_01( dims(1:2), winSize );
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
    case 2 % heat kernel weighting:
        [ri, ci, val] = mexWeightCD_Gauss(data, dims, winSize, sigma);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
    case 3 % Dot-Product Weighting
        [ri, ci, val] = mexWeightCD_Sad(data, dims, winSize);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
        
        %% neighbour percent
    case 4 % sad neighbour percent
        [ri, ci, val] = mexWeightSad_neighbour(data, dims, winSize, percent);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
    case 5 % corr neighbour percent
        [ri, ci, val] = mexWeightCorr_neighbour(data, dims, winSize, percent);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
        
        %% cluster percent
    case 6 % sad cluster by kmeans    percent
        [ri, ci, val] = mexWeightSad_cluster(data, dims, IDX, winSize, percent);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
    case 7 % corr cluster by kmeans   percent
        [ri, ci, val] = mexWeightCorr_cluster(data, dims, IDX, winSize, percent);
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
        
        %% turbepixle percent
    case 8 % corr cluster by turbopixel   percent
        W = sparse(nSmp, nSmp);
        for ii = 1:size(IDX, 2)
            [ri, ci, val] = mexWeightCorr_tuborpixel(data, dims, IDX(:, ii), winSize, percent);
            W = W + 0.5^(ii-1)*sparse(ri+1, ci+1, val, nSmp, nSmp);
        end
        
        %% weight defined by others
    case 9 % the weight matrix for WNMF, ref: Enhancing Spectral Unmixing by Local Neighborhood Weights
        [ri, ci, val] = mexWeightforWNMF (data, dims, winSize);
        W = sparse (ri+1, ci+1, val, nSmp, nSmp);
        
    otherwise
        [ri, ci, val] = mexWeightMatrix_gauss(reshape(data', dims), winSize, sigma );
        W = sparse(ri+1, ci+1, val, nSmp, nSmp);
        %         s = message('weightType must be [1, 8]');
        %         warning(s);
end
