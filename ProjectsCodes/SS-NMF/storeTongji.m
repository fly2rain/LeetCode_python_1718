% noAlg =       1       2      3        4           5       6          7        8
algNames = {'SS-NMF', 'VCA', 'NMF', 'L_1-NMF', 'L1_2-NMF', 'G-NMF', 'W-NMF', 'EDC-NMF'};
NO_Alg   = length(algNames);

if exist('cellResObj.mat', 'file')
    load cellResObj.mat;
else
    cellResObj = cell(NO_Alg, 3);
end
cellResObj{noAlg, 1} = algNames{noAlg};
cellResObj{noAlg, 2} = resObjMean;
cellResObj{noAlg, 3} = resObjStd;
save('cellResObj.mat', 'cellResObj');