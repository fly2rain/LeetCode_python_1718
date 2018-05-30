function [U_final, V_final, nIter_final, elapse_final, bSuccess, objhistory_final] = GNMF_KL(X, k, W, options, U, V)
% Graph regularized Non-negative Matrix Factorization with Divergence Formulation 
% Locality Preserving Non-negative Matrix Factorization (LPNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of words (vocabulary size)
%       nSmp  ... number of documents
% k ... number of hidden factors
% W ... weight matrix of the affinity graph 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
% References:
% [1] Deng Cai, Xiaofei He, Xuanhui Wang, Hujun Bao, and Jiawei Han.
% "Locality Preserving Nonnegative Matrix Factorization", Proc. 2009 Int.
% Joint Conf. on Arti_cial Intelligence (IJCAI-09), Pasadena, CA, July 2009. 
%
% [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
% Non-negative Matrix Factorization for Data Representation", IEEE
% Transactions on Pattern Analysis and Machine Intelligence, to appear.
%
%
%
%   version 2.0 --April/2009 
%   version 1.0 --April/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


ZERO_OFFSET = 1e-200;


differror = 1e-5;
if isfield(options,'error')
    differror = options.error;
end


maxIter = [];
if isfield(options, 'maxIter')
    maxIter = options.maxIter;
end


nRepeat = 10;
if isfield(options,'nRepeat')
    nRepeat = options.nRepeat;
end

minIterOrig = 30;
if isfield(options,'minIter')
    minIterOrig = options.minIter;
end
minIter = minIterOrig-1;

meanFitRatio = 0.1;
if isfield(options,'meanFitRatio')
    meanFitRatio = options.meanFitRatio;
end

if ~isfield(options,'InitWay')
    options.InitWay = 'random';
end

alpha = 1;
if isfield(options,'alpha')
    alpha = options.alpha;
end


Norm = 2;
NormV = 1;

if min(min(X)) < 0 
    error('Input should be nonnegative!');
end

[mFea,nSmp]=size(X); 
NCWeight = [];
if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    feaSum = full(sum(X,2));
    NCWeight = (X'*feaSum).^-1;
    tmpNCWeight = NCWeight;
else
    tmpNCWeight = ones(nSmp,1);
end

if issparse(X)
    nz = nnz(X);
    nzk = nz*k;
    [idx,jdx,vdx] = find(X);
    if isempty(NCWeight)
        Cons = sum(vdx.*log(vdx) - vdx);
    else
        Cons = sum(NCWeight(jdx).*(vdx.*log(vdx) - vdx));
    end
    ldx = sub2ind(size(X),idx,jdx);
else
    Y = X + ZERO_OFFSET;
    if isempty(NCWeight)
        Cons = sum(sum(Y.*log(Y)-Y));
    else
        Cons = sum(NCWeight'.*sum(Y.*log(Y)-Y,1));
    end
    clear Y;
end

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    alpha = alpha*nSmp;    
end

Method = 1; % Multiplicative Update 
if isfield(options,'OptimizeMethod')
    if strcmpi(options.OptimizeMethod,'MultipUpdate')
        Method = 1; % Multiplicative Update
    elseif strcmpi(options.OptimizeMethod,'ProjGrad')
        error('not implemented!');
        Method = 2; % Projective Gradient
    else
        error('not implemented!');
        Method = 3; % Multiplicative Update + Projective Gradient
    end
end
realMethod = Method;


bSuccess = 1;


if ~exist('U','var')
    U = abs(rand(mFea,k));
    V = abs(rand(nSmp,k));
else
    nRepeat = 1;
end
[U,V] = NormalizeUV(U, V, NormV, Norm);


DCol = full(sum(W,2));
D = spdiags(DCol,0,speye(size(W,1)));
L = D - W;
if isfield(options,'NormW') && options.NormW
    D_mhalf = DCol.^-.5;

    tmpD_mhalf = repmat(D_mhalf,1,nSmp);
    L = (tmpD_mhalf.*L).*tmpD_mhalf';
    clear D_mhalf tmpD_mhalf;
end

L = alpha*L;
L = max(L, L');



% if nRepeat == 1
%     if issparse(X)
%         [obj_NMFhistory, obj_Laphistory] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
%     else
%         [obj_NMFhistory, obj_Laphistory] = CalculateObj(Cons, X, U, V, L, NCWeight);
%     end
%     objhistory = obj_NMFhistory + obj_Laphistory;
% %     meanFit = objhistory*10;
% %     minIterOrig = 0;
% %     minIter = 0;
% end

maxM = 62500000;
mn = numel(X);
if issparse(X)
    nBlockNZ = floor(maxM/(k*2));
end

tryNo = 0;
selectInit = 1;
% if nRepeat == 1
%     selectInit = 0;
% end
while tryNo < nRepeat
    tmp_T = cputime;
    tryNo = tryNo+1;
    nIter = 0;
    maxErr = 1;
    while(maxErr > differror)
        if Method == 1
            %tmpTTT = cputime;
            if issparse(X)
                % ===================== update V ========================
                if nzk < maxM    
                    Y = sum(U(idx,:).*V(jdx,:),2);
                else
                    Y = zeros(size(vdx));
                    for i = 1:ceil(nz/nBlockNZ)
                        if i == ceil(nz/nBlockNZ)
                            smpIdx = (i-1)*nBlockNZ+1:nz;
                        else
                            smpIdx = (i-1)*nBlockNZ+1:i*nBlockNZ;
                        end
                        Y(smpIdx) = sum(U(idx(smpIdx),:).*V(jdx(smpIdx),:),2);
                    end
                end                    
                Y = vdx./max(Y,1e-10);
                Y = sparse(idx,jdx,Y,mFea,nSmp);
                Y = (U'*Y)';

                if isempty(NCWeight)
                    Y = V.*Y;
                else
                    Y = repmat(NCWeight,1,k).*V.*Y;
                end
                sumU = max(sum(U,1),1e-10);
                
                for i = 1:k
                    tmpL = L;
                    for j = 1:nSmp
                        tmpL(j,j) = tmpL(j,j) + tmpNCWeight(j)*sumU(i);
                    end
                    %V(:,i) = tmpL\Y(:,i);
                    [R,p] = chol(tmpL);
                    if p == 0
                        V(:,i) = R\(R'\Y(:,i));
                    end
                end
                
                % ===================== update U ========================
                if nzk < maxM
                    Y = sum(U(idx,:).*V(jdx,:),2);
                else
                    Y = zeros(size(vdx));
                    for i = 1:ceil(nz/nBlockNZ)
                        if i == ceil(nz/nBlockNZ)
                            smpIdx = (i-1)*nBlockNZ+1:nz;
                        else
                            smpIdx = (i-1)*nBlockNZ+1:i*nBlockNZ;
                        end
                        Y(smpIdx) = sum(U(idx(smpIdx),:).*V(jdx(smpIdx),:),2);
                    end
                end
                
                Y = vdx./max(Y,1e-10);
                Y = sparse(idx,jdx,Y,mFea,nSmp);
                if isempty(NCWeight)
                    Y = Y*V;
                    sumV = max(sum(V,1),1e-10);
                else
                    wV = repmat(NCWeight,1,k).*V;
                    Y = Y*wV;
                    sumV = max(sum(wV,1),1e-10);
                    clear wV;
                end
                U = U.*(Y./repmat(sumV,mFea,1));
            else
                if mn < maxM
                    % ===================== update V ========================
                    Y = U*V';
                    Y = X./max(Y,1e-10);
                    Y = Y'*U;

                    if isempty(NCWeight)
                        Y = V.*Y;
                    else
                        Y = repmat(NCWeight,1,k).*V.*Y;
                    end
                    sumU = max(sum(U,1),1e-10);
                    
                    for i = 1:k
                        tmpL = L;
                        for j = 1:nSmp
                            tmpL(j,j) = tmpL(j,j) + tmpNCWeight(j)*sumU(i);
                        end
                        %V(:,i) = tmpL\Y(:,i);
                        [R,p] = chol(tmpL);
                        if p == 0
                            V(:,i) = R\(R'\Y(:,i));
                        end
                    end
                    
                    % ===================== update U ========================
                    Y = U*V';
                    Y = X./max(Y,1e-10);
                    if isempty(NCWeight)
                        Y = Y*V;
                        sumV = max(sum(V,1),1e-10);
                    else
                        wV = repmat(NCWeight,1,k).*V;
                        Y = Y*wV;
                        sumV = max(sum(wV,1),1e-10);
                        clear wV;
                    end
                    U = U.*(Y./repmat(sumV,mFea,1));
                else
                    error('Not implemented!');
                end
            end
            %Time_Update = cputime - tmpTTT;
            clear Y sumU sumV;
            

            nIter = nIter + 1;
            if nIter > minIter
                if selectInit
                    %tmpTTT = cputime;
                    if issparse(X)
                        [obj_NMFhistory, obj_Laphistory] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
                    else
                        [obj_NMFhistory, obj_Laphistory] = CalculateObj(Cons, X, U, V, L, NCWeight);
                    end
                    %Time_Obj = cputime - tmpTTT;
                    objhistory = obj_NMFhistory + obj_Laphistory;
                    maxErr = 0;
                else
                    if isempty(maxIter)
                        %tmpTTT = cputime;
                        if issparse(X)
                            [obj_NMF, obj_Lap] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight);
                        else
                            [obj_NMF, obj_Lap] = CalculateObj(Cons, X, U, V, L, NCWeight);
                        end
                        %Time_Obj = [Time_Obj; cputime - tmpTTT];
                        newobj = obj_NMF + obj_Lap;
                        
                        obj_NMFhistory = [obj_NMFhistory obj_NMF]; %#ok<AGROW>
                        obj_Laphistory = [obj_Laphistory obj_Lap]; %#ok<AGROW>
                        
                        objhistory = [objhistory newobj]; %#ok<AGROW>
                        meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                        maxErr = (meanFit-newobj)/meanFit;
                    else
                        maxErr = 1;
                        if nIter >= maxIter
                            maxErr = 0;
                            objhistory = 0;
                        end
                    end
                end
            end
        else
            error('not implemented!');
        end
    end

    if maxErr < 0
        warning('Iterative Procedure Error!');
        bSuccess = 0;
    end
    
    elapse = cputime - tmp_T;
    Method = realMethod;

    if tryNo == 1
        U_final = U;
        V_final = V;
        nIter_final = nIter;
        elapse_final = elapse;
        objhistory_final = objhistory;
    else
        if objhistory(end) < objhistory_final(end)
            U_final = U;
            V_final = V;
            nIter_final = nIter;
            objhistory_final = objhistory;
            if selectInit
                elapse_final = elapse;
            else
                elapse_final = elapse_final+elapse;
            end
        end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea,k));
            V = abs(rand(nSmp,k));
            
            [U,V] = NormalizeUV(U, V, NormV, Norm);
        else
            tryNo = tryNo - 1;
            minIter = 0;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory*10;
        end
    end
end

nIter_final = nIter_final + minIterOrig;

if isfield(options,'NormU') && options.NormU
    NormV = 0;
end
if isfield(options,'Norm') 
    Norm = options.Norm;
end

[U_final,V_final] = NormalizeUV(U_final, V_final, NormV, Norm);


function [obj_NMF, obj_Lap] = CalculateObjSparse(Cons, jdx, vdx, ldx, U, V, L, NCWeight)
    ZERO_OFFSET = 1e-200;
    
    maxM = 62500000;
    mFea = size(U,1);
    nSmp = size(V,1);
    mn = mFea*nSmp;
    nBlock = floor(maxM/(mFea*2));

    if mn < maxM
        Y = U*V';
        if isempty(NCWeight)
            obj_NMF = sum(sum(Y)) - sum(vdx.*log(Y(ldx)+ZERO_OFFSET)); % 14p (p << mn)
        else
            obj_NMF = sum(NCWeight'.*sum(Y,1)) - sum(NCWeight(jdx).*(vdx.*log(Y(ldx)+ZERO_OFFSET))); % 14p (p << mn)
        end
    else
        obj_NMF = 0;
        ldxRemain = ldx;
        vdxStart = 1;
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
                ldxNow = ldxRemain;
                vdxNow = vdxStart:length(vdx);
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
                ldxLast = find(ldxRemain <= mFea*i*nBlock, 1 ,'last');
                ldxNow = ldxRemain(1:ldxLast);
                ldxRemain = ldxRemain(ldxLast+1:end);
                vdxNow = vdxStart:(vdxStart+ldxLast-1);
                vdxStart = vdxStart+ldxLast;
            end
            Y = U*V(smpIdx,:)';
            if isempty(NCWeight)
                obj_NMF = obj_NMF + sum(sum(Y)) - sum(vdx(vdxNow).*log(Y(ldxNow-mFea*(i-1)*nBlock)+ZERO_OFFSET));
            else
                obj_NMF = obj_NMF + sum(NCWeight(smpIdx)'.*sum(Y,1)) - sum(NCWeight(jdx(vdxNow)).*(vdx(vdxNow).*log(Y(ldxNow-mFea*(i-1)*nBlock)+ZERO_OFFSET)));
            end
        end
    end
    obj_NMF = obj_NMF + Cons;    

    Y = log(V + ZERO_OFFSET);
    obj_Lap = sum(sum((L*Y).*V));


function [obj_NMF, obj_Lap] = CalculateObj(Cons, X, U, V, L, NCWeight)
    ZERO_OFFSET = 1e-200;
    [mFea, nSmp] = size(X);
    maxM = 62500000;
    mn = numel(X);
    nBlock = floor(maxM/(mFea*2));

    if mn < maxM
        Y = U*V'+ZERO_OFFSET;
        if isempty(NCWeight)
            obj_NMF = sum(sum(Y - X.*log(Y))); % 14mn
        else
            obj_NMF = sum(NCWeight'.*sum(Y - X.*log(Y),1)); % 14mn
        end
    else
        obj_NMF = 0;
        for i = 1:ceil(nSmp/nBlock)
            if i == ceil(nSmp/nBlock)
                smpIdx = (i-1)*nBlock+1:nSmp;
            else
                smpIdx = (i-1)*nBlock+1:i*nBlock;
            end
            Y = U*V(smpIdx,:)'+ZERO_OFFSET;
            if isempty(NCWeight)
                obj_NMF = obj_NMF + sum(sum(Y - X(:,smpIdx).*log(Y)));
            else
                obj_NMF = obj_NMF + sum(NCWeight(smpIdx)'.*sum(Y - X(:,smpIdx).*log(Y),1));
            end
        end
    end
    obj_NMF = obj_NMF + Cons;
    
    Y = log(V + ZERO_OFFSET);
    obj_Lap = sum(sum((L*Y).*V));


function [U, V] = NormalizeUV(U, V, NormV, Norm)
    nSmp = size(V,1);
    mFea = size(U,1);
    if Norm == 2
        if NormV
            norms = sqrt(sum(V.^2,1));
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sqrt(sum(U.^2,1));
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    else
        if NormV
            norms = sum(abs(V),1);
            norms = max(norms,1e-10);
            V = V./repmat(norms,nSmp,1);
            U = U.*repmat(norms,mFea,1);
        else
            norms = sum(abs(U),1);
            norms = max(norms,1e-10);
            U = U./repmat(norms,mFea,1);
            V = V.*repmat(norms,nSmp,1);
        end
    end