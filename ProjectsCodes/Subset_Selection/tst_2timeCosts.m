clc, clear all;

NTry    = 10;
nFea    = 200;
% NSmpGr  = [1000, 8000, 20000, 30000];
NSmpGr  = [1000, 5000, 8000, 10000, 15000, 20000, 25000, 30000, 35000];

% NSmpGr  = [200, 200, 200, 200];

NGr = length (NSmpGr);
StmCell  = cell (NGr, 3);
Stm1     = zeros (NGr, 3);
Stm2     = zeros (NGr, 3);
for ii = 1 : NGr
    StmCell{ii, 1} = NSmpGr (ii);
end

s1tm = zeros (NTry + 1, 3);
s2tm = zeros (NTry + 1, 3);

for nn = 1 : NGr
    nSmp = NSmpGr (nn);
    
    %% set 1 =========== column by column linear equation  vs. all togethter equation ==
    a = rand (nSmp, nFea);
    A = a.' * a + 0.01*eye (nFea);
    C = zeros (nFea, nSmp);
    B = rand (nFea, nSmp);
    
    for ii = 1 : NTry
        tic,
        for j = 1:nSmp,
            b = B (:, j);
            C (:, j) = A \ b;
        end,
        s1tm (ii, 1) = toc;
        
        tic, D = A \ B;
        s1tm (ii, 2) = toc;
        
        s1tm (ii, 3) = s1tm (ii, 1) / s1tm (ii, 2);
    end
    s1tm (NTry+1, :) = mean (s1tm, 1);
    StmCell{nn, 2} = s1tm;
    Stm1 (nn, :) = mean (s1tm, 1);
    
    %% set 2 =========== column by column linear equation  vs. all togethter equation ==
    f = rand (nFea, 1);
    e = rand (nSmp, 1);
    Af = a.' * a + 0.01*eye (nFea);
    Ae = a * a.' + 0.01*eye (nSmp);
    
    for j = 1 : NTry
        tic, ae = Ae \ e; s2tm (j, 1) = toc;
        tic, af = Af \ f; s2tm (j, 2) = toc;
        
        s2tm (j, 3) = s2tm (j, 1) / s2tm (j, 2);
    end
    
    s2tm (NTry+1, :) = mean (s2tm, 1);
    StmCell{nn, 3} = s2tm;
    Stm2 (nn, :) = mean (s2tm, 1);
end
save ('timeStatistics.mat', 'Stm1', 'Stm2', 'StmCell');
Stm = cat ();
% %% test 3 =======================
% W = rand(2);
% W = W ./ repmat (sum(W,1), [2 1]);
% lambda = rand (2, 1);
% lambda = lambda ./ sum (lambda);
% Lambda = diag(lambda);
%
% Lz = Lambda + W*Lambda*W.';
% Lf = Lambda*W.' + W*Lambda;
%
% rst = 2*Lz - Lf
