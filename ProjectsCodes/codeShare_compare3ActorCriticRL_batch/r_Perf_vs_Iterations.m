% learn the adaptive intervention on 06/04/2016
clear all;
addpath ('Data_RL_algs\');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GrAlgOPT = {'fmincon', 'fminconGrad'};
% GrGamma = [0, 0.5, 0.9];  NGamma=length(GrGamma); alpha=1;
% GrGamma = [0, 0.2, 0.5, 0.9, 0.99];  NGamma=length(GrGamma); alpha=0;
GrGamma = [0, 0.3, 0.7, 0.95, 1];  NGamma=length(GrGamma);
GrMaxIter = [10, 20, 50, 80, 120]; NMaxIter = length (GrMaxIter);

%% data #6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%% the parameter we use, 05/12/2016 %%%%%%%%%%
% % % T = 200;
% datIdx=6; nbeta=4; NPeo=20; T=200; md.Lo=10; maxIter=30; nseRwd=1; nseSt=1; 
% OptAlg=GrAlgOPT{2}; dirName=''; rngVal=10; valBnd=10; zetaA=0.05; alpha=1;
% zetaC=0.05; order=1; FoChoice='itself';

% % T = 200, NPeo=50
datIdx=6; nbeta=4; NPeo=50; T=200; md.Lo=10; maxIter=30; nseRwd=1; nseSt=1; 
OptAlg=GrAlgOPT{2}; dirName=''; rngVal=10; valBnd=10; zetaA=0.05; alpha=1;
zetaC=0.05; order=1; FoChoice='itself';


% the betas for dataset #1 --
GrBeta1 = [
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,10,0.25,0.25,0.4,1,0.4;   %1 Susan's beta
    0.3,0.2,0.8,0.3,0.05,0.9,0.25,10,0.15,0.15,0.95,0.5,0.2; %2 low teatment fatigure
    0.3,0.2,0.8,0.3,0.05,0.9,0.25,10,0.15,0.15,0.95,0.5,0.8; %3 high treatment fatigure
    0.3,0.2,0.8,0.3,0.05,0.9,0.25,10,0.15,0.15,0.25,0.5,0.8; %4 very high treatment fatigure
    0.3,0.2,0.8,0.3,0.05,0.9,0.25,10,0.15,0.15,0.95,0.5,0.5; %5 mid teatment fatigure
    0.3,0.2,0.8,0.3,0.05,0.9,0.25,10,0.15,0.15,0.95,0.5,0.4; %6 susan's setting of teatment fatigure
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,10,0.25,0.25,0.4,0.1,0.9; %7 only one different element with Susan's beta
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,10,0.25,0.25,0.4,0.1,0.5;  %8
    ]; % only useful on

% the betas for dataset #5 --
GrBeta5 = [
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,10,0.25,0.25,0.4,0.1,0.9, 0, 0, 0, 0;%1 only one different element with Susan's beta
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,10,0.25,0.25,0.4,0.1,0.9, 1, 2, 1, 1;%2
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,10,0.25,0.25,0.4,0.1,0.4, 0, 0, 0, 0;%3
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,00,0.25,0.25,0.4,0.1,0.9, 0, 0, 0, 0;%4 only one different element w
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,00,0.25,0.25,0.4,0.1,0.4, 0, 0, 0, 0;%5 only one different element w
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,00,0.25,0.25,0.4,0.1,1.5, 0, 0, 0, 0;%6 only one different element w
    0.4,0.25,0.8,0.5,0.05,0.5,0.25,00,0.25,0.25,0.4,0.1,1.5, 0.2, 0.4, 0.5, 0.8;%7 only one different element w
    0.4,0.25,0.8,0.7,0.05,0.5,0.25,3,0.25,0.25,0.4,0.1,1.5, 0, 0, 0, 0;%8 only one different element w
    0.4,0.25,0.8,0.7,0.05,0.5,0.25,3,0.25,0.25,0.4,0.1,1.5, 0, 0, 0, 0;%9 only one different element w
    ]; % only useful on

% the betas for dataset #1 --
GrBeta6 = [
    0.4,0.25,0.8,0.7,0.05,0.5,0.25,3,0.25,0.25,0.4,0.1,1.5, 0, 0, 0, 0, 500;%1 only one different element w
    0.4,0.25,0.8,0.7,0.05,0.5,0.25,2,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 500;%2
    0.4,0.3,0.4,0.7,0.2,0.8,0.25,2,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 500;%3
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,3,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 500;%4
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,3,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 1, 500;%5
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,1,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 500;%6
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,10,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 1;%7
    0.4,0.3,0.4,0.7,0.05,0.6,0.25,10,0.25,0.25,0.4,0.1,0.5, 0, 0, 0, 0, 500;%8
    ]; % only useful on

if datIdx == 1
    beta = GrBeta1 (nbeta,:).';
elseif datIdx == 2
    beta = GrBeta2 (nbeta,:).';
elseif datIdx == 3
    beta = GrBeta3 (nbeta,:).';
elseif datIdx == 5
    beta = GrBeta5 (nbeta,:).';
elseif datIdx == 6
    beta = GrBeta6 (nbeta,:).';
else
    beta = [];
end

% 1) polynomial, 2) RBF,  3) sparse
md.Lgo = md.Lo + 1;
md.FoChoice = FoChoice;
md.order = order;
switch md.FoChoice
    case 'itself'
        md.Lfo = md.Lo;
    case 'polynomial'
        md.Lfo = (md.order+1)^md.Lo;
    case 'RBF'
        md.Lfo = md.order^md.Lo;
    case 'sparse'
        md.Lfo = md.Lov^md.Lo;
end
md.Lho = 2*md.Lfo + 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% #2 train & test the methods ========================================
endTm   = 5e3; % end   time for long run average measure
starTm  = 1e3; % start time for long run average measure

%% set the file name to store the different results -----
rstName = [dirName 'PvsMaxIter AI-Batch datIdx=' num2str(datIdx) ' nbeta=' num2str(nbeta) ...
    ' ' OptAlg ' T=' num2str(T) ' alpha=' num2str(alpha) ' NPeo=' num2str(NPeo) ...
    ' Lo=' num2str(md.Lo) ' NFea=' num2str(md.Lho) ' nseRwd=' num2str(nseRwd) ...
    ' nseSt=' num2str(nseSt) ' zetaC=' num2str(zetaC) ' zetaA=' num2str(zetaA) '.mat'];

% create the matrices to store the results --
GrTheta = zeros( md.Lgo, NPeo, NGamma, NMaxIter);
GrW     = zeros( md.Lho, NPeo, NGamma, NMaxIter);
% (1,1) meanVal, (1,2) stdVal, (2,1) meanPol, (2,2) stdPol
tsRstCell  = cell (2,2);
meanPolicy = zeros (NGamma, NMaxIter);
stdPolicy = zeros (NGamma, NMaxIter);
meanValue = zeros (NGamma, NMaxIter);
stdValue  = zeros (NGamma, NMaxIter);

sqrtNPeo = sqrt (NPeo);
%  id not exists, run for the results
if ~exist (rstName, 'file')
    for nm = 1 : NMaxIter
        tic;
        maxIter = GrMaxIter (nm);
        
        % generate the data to compare the 3 methods in the Adaptive intervention setting.
        [GrX, GrY, GrA, GrR]  = F_prepareDat_cmp3methods ('datIdx',datIdx,'TT',T, ...
            'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,'noiseStat',nseSt, ...
            'beta',beta,'rngVal',rngVal,'NPeo',NPeo);
        
        for np = 1 : NPeo
            % re-organize the data for each individual --------------------
            aaa = GrA (:, np);
            rrr = GrR (:, np);
            X   = GrX (:, :, np);
            Y   = GrY (:, :, np);            
            % train the model ---------------------------------------------
            parfor ng = 1 : NGamma
                rng (rngVal, 'twister');
                gamma = GrGamma(ng);
                [theta, w] = F_batchRL (aaa, rrr, X, Y, 'gamma',gamma,'maxIter',maxIter, ...
                    'mydim',md,'zetaC',zetaC,'zetaA',zetaA, ...
                    'OptAlg',OptAlg,'valBnd',valBnd );
                GrTheta(:, np, ng, nm) = theta;
                GrW(:, np, ng, nm) = w;
            end
        end
        % test the results ---------------------------------------------
        rng (rngVal, 'twister');
        GrO0Ts = F_simuInitStates (md, NPeo, datIdx);
        parfor ng = 1 : NGamma
            rng (rngVal, 'twister');
            gamma = GrGamma(ng);
            thetaN = GrTheta (:,:,ng,nm);
            wN     = GrW (:,:,ng,nm);
            [NLongRwd, rmseVal] = F_tstLongAvgRwd_val (GrO0Ts,thetaN,wN,...
                'datIdx',datIdx,'EndTime',endTm,'StarTime',starTm,'gamma',gamma,...
                'alpha', alpha,'mydim',md,'nseRwd',nseRwd,'nseSt',nseRwd, ...
                'beta',beta);
            
            meanPolicy (ng, nm) = mean (NLongRwd);
            stdPolicy  (ng, nm) = std (NLongRwd) ./ sqrtNPeo * 2;
            meanValue(ng, nm) = mean (rmseVal);
            stdValue (ng, nm) = std (rmseVal) ./ sqrtNPeo * 2;
        end
        algTime = toc / 60;
        fprintf ('[%d]\talpha:%.3f\ttime:%.2f\n',nm,alpha,algTime);
    end
    
    tsRstCell{1,1} = meanPolicy;
    tsRstCell{1,2} = stdPolicy;
    tsRstCell{2,1} = meanValue;
    tsRstCell{2,2} = stdValue;
    
    save ( rstName, 'tsRstCell');
else
    load ( rstName );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% #3 show the results  =======================================
% stdVal = zeros (NGamma, NEpsilon);
cmpItem = '# iteration';
sevenColors = {'-<m', '-dc', '-*g', '-or', '-^b', '-sk', '->y'};
figure (1)
set(gcf, 'Unit','centimeters');
set(gcf, 'Position',[0,2,32,10]);
% fileOptAvgRwd = 'optimal average reward for data#2.mat';
% load (fileOptAvgRwd);
% get the percent performance divided by the opimal policy.
% RstAvgRwd_Percent = RstAvgRwd ./ repmat(optAvgRwdTheory, 1, NGamma) * 100;
nameOfMethod = cell(NGamma, 1);
% plot the long term average reward in an absolute scale
subplot (1,2,1);
for ng = 1 : NGamma
    nameOfMethod{ng} =  ['\gamma=' num2str(GrGamma(ng))];
    iAvgRwd = tsRstCell{1,1}(ng, :);
    iStdRwd = tsRstCell{1,2}(ng, :); % / sqrt(NPeoTs);
    %     plot (GrAlpha, iAvgRwd, sevenColors{ng}, ...
    %         'LineWidth', 2, 'MarkerSize', 6);
    errorbar (GrMaxIter, iAvgRwd, iStdRwd, sevenColors{ng}, ...
        'LineWidth', 2, 'MarkerSize', 6);
    hold on;
end
xlim ([5, 125]);
set(0,'DefaultTextFontname', 'Times New Roman');
xlabel (cmpItem, 'FontSize', 14);
ylabel ('Expect. Long run average reward', 'FontSize', 14)
title (['Policy Performance vs. ' cmpItem], 'FontSize', 14);
legend (nameOfMethod, 'FontSize', 12,  'Location','northwest');
grid on;

%  plot the long term average reward as a percentage of the optimal reward.
subplot (1,2,2);
for ng = 1 : NGamma
    nameOfMethod{ng} =  ['\gamma=' num2str(GrGamma(ng))];
    iAvgRwd = tsRstCell{2,1}(ng, :);
    iStdRwd = tsRstCell{2,2}(ng, :); % / sqrt(NPeoTs);
    %      plot (GrAlpha, iAvgRwd, sevenColors{ng}, ...
    %         'LineWidth', 2, 'MarkerSize', 6);
    errorbar (GrMaxIter, iAvgRwd, iStdRwd, sevenColors{ng}, ...
        'LineWidth', 2, 'MarkerSize', 6);
    hold on;
end
xlim ([5, 125]);
set(0,'DefaultTextFontname', 'Times New Roman');
xlabel (cmpItem, 'FontSize', 14);
ylabel ('RMSE between Q_{MC} and Q_{linear approx.} ', 'FontSize', 14)
title ( ['Value Approximate Error vs. ' cmpItem], 'FontSize', 14);
legend (nameOfMethod, 'FontSize', 12,  'Location','northwest');
grid on;