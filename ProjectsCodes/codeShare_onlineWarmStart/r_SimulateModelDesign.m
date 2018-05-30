% design a simulation model that meets our requirement
clear all;
addpath ('Data_RL_algs\')
%% #1 parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
endTm    = 5e3; % end   time for long run average measure
starTm   = 1e3; % start time for long run average measure

% verion  @ 08/01/2016 longer trajectory is involved, T=500 ---------------------
GrSigmaBt = [0, 0.001, 0.01, 0.05]; NSigmaBt = length (GrSigmaBt);
GrRndPy   = [0, 0.25, 0.5, 0.75, 1]; NRndPy = length (GrRndPy);
GrSigmaBtName  = cell (NSigmaBt, 1);
for i = 1 : NSigmaBt
    GrSigmaBtName{i} = [ 'nseBt:' num2str(GrSigmaBt(i)) ];
end

datIdx=6; md.Lo=3; nseRwd=1; nseSt=1; T0=42; order=1; FoChoice='itself';
alpha=1; dirName='v1_'; NPeoBC=40; NPeoOL=40; %%%%%%

nbeta=1; rngVal = 20;
% nbeta=2; rngVal=20;
% nbeta=2; rngVal=40; % **
% nbeta=2; rngVal=80;
% nbeta=2; rngVal=100;

% nbeta=3; rngVal=40; % **
% nbeta=3; rngVal=10; % **

% nbeta=4; rngVal=40;

% nbeta=5; rngVal=40;
% nbeta=6; rngVal=40;
% nbeta=7; rngVal=40;

% nbeta=8; rngVal=40;
% nbeta=8; rngVal=20;

% nbeta=9; rngVal=20; dirName='v1_';
% nbeta=10; rngVal=20;

datIdx=6; nbeta=12; rngVal=20;

GrBeta6 = [
    0.40,0.25,0.35,0.65,0.10,0.50,0.22,2.00,0.15,0.20,0.32,0.10,0.45,0.00,0.00,0.00,0.00,5; % 1
    0.40,0.25,0.35,0.45,0.10,0.30,0.22,2.00,0.15,0.20,0.12,0.10,0.15,0.00,0.00,0.00,0.00,5; % 2
    0.40,0.25,0.30,0.30,0.15,0.25,0.25,2.00,0.15,0.20,0.15,0.10,0.40,0.00,0.00,0.00,0.00,5; % 3
    0.40,0.25,0.50,0.30,0.15,0.25,0.25,2.00,0.15,0.30,0.15,0.10,0.40,0.00,0.00,0.00,0.00,5; % 4
    0.40,0.25,0.25,0.30,0.15,0.25,0.25,2.00,0.15,0.30,0.15,0.10,0.40,0.00,0.00,0.00,0.00,5; % 5
    0.40,0.25,0.35,0.35,0.10,0.50,0.22,2.00,0.15,0.20,0.32,0.10,0.45,0.00,0.00,0.00,0.00,5; % 6
    0.40,0.25,0.35,0.35,0.10,0.50,0.22,2.00,0.15,0.20,0.32,0.10,0.20,0.00,0.00,0.00,0.00,5; % 7
    0.40,0.25,0.60,0.35,0.10,0.20,0.22,2.00,0.15,0.20,0.65,0.10,0.15,0.00,0.00,0.00,0.00,5; % 8
    0.40,0.25,0.90,0.35,0.10,0.20,0.22,2.00,0.15,0.20,0.95,0.10,0.15,0.00,0.00,0.00,0.00,5; % 9
    0.40,0.25,0.35,0.65,0.05,0.20,0.22,2.00,0.15,0.20,0.32,0.10,0.10,0.00,0.00,0.00,0.00,5; % 10
    0.40,0.25,0.35,0.32,0.05,0.20,0.22,2.00,0.15,0.20,0.32,0.10,0.10,0.00,0.00,0.00,0.00,5; % 11
    0.40,0.55,0.35,0.32,0.05,0.20,0.22,2.00,0.15,0.20,0.32,0.10,0.10,0.00,0.00,0.00,0.00,5; % 12
    ];


beta = GrBeta6(nbeta, :);

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

% define the filename for the results, OL Online
rstName = [dirName 'simulateDesign datIdx=' num2str(datIdx) ' nbeta=' num2str(nbeta) ...
    ' NPeoOL=' num2str(NPeoOL) ' alpha=' num2str(alpha) ' Lo=' num2str(md.Lo) ...
    ' nseRwd=' num2str(nseRwd) ' nseSt=' num2str(nseSt) ' rngVal=' num2str(rngVal) '.mat'];

%% create the matrices to store the results --W
avgPolicy = zeros ( NRndPy, NSigmaBt );
stdPolicy = zeros ( NRndPy, NSigmaBt );

T0_1    = T0 - 1;
GrSeed  = ones(100, 1) * rngVal;
%% #2 run and test the algorithms
%  id not exists, run for the results
% if ~exist (rstName, 'file')
for nb = 1 : NSigmaBt
    nseBt = GrSigmaBt (nb);
    %% generate the data
    [GrBetaBC, GrBetaOL]  = F_prepareDat_warmStart (...
        'datIdx',datIdx,'TT',T0,'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,...
        'noiseStat',nseSt,'noiseBeta',nseBt,'Beta',beta,'rngVal',rngVal,...
        'NPeoBC',NPeoBC,'NPeoOL',NPeoOL);
    
    rng (GrSeed(nb));
    GrO0 = F_simuInitStates (md, NPeoOL, datIdx);
    for nm = 1 : NRndPy
        rndPy = GrRndPy (nm);
        rng (GrSeed(nm));
        NLongRwd = F_tstLongAvgRwd_design_4Each (GrO0,rndPy,...
            'datIdx',datIdx,'EndTime',endTm,'StarTime',starTm,...
            'alpha', alpha,'nseRwd',nseRwd,'nseSt',nseSt, ...
            'GrBeta',GrBetaOL);
        
        avgPolicy (nm, nb) = mean(NLongRwd);
        stdPolicy (nm, nb) = std (NLongRwd) ./ sqrt(NPeoOL) * 2;
        % GrW(:, np, nt, nm, ng, nb) = w;
    end
end
%     save ( rstName, 'avgPolicy', 'stdPolicy');
% else
%     load ( rstName );
% end

%% #3 organize the results ------------- NBA end 4-------------
cmpItem = 'the probability of providing treatment';
sevenColors = {'-<b', '-dr', '-*k', '-om', '-^c', '-sg', '->y'};

xlimRange = [-0.1, 1.1];
figure (1)
for nb = 1 : NSigmaBt
    iAvgRwd = avgPolicy (:, nb);
    iStdRwd = stdPolicy (:, nb); % / sqrt(NPeoTs);
    %         iStdRwd = zeros (NT, 1);
    errorbar (GrRndPy, iAvgRwd, iStdRwd, sevenColors{nb}, ...
        'LineWidth', 2, 'MarkerSize', 6);
    hold on;
end
% set the figure property
xlim ( xlimRange );
% ylim auto;
% ylim ( [6.5, 10.3] );
set(0,'DefaultTextFontname', 'Times New Roman');
xlabel (cmpItem, 'FontSize', 14);
ylabel ('Expect. Long run average reward', 'FontSize', 14)
title (['Perform. vs. nseBt and probability of giving treatment'], 'FontSize', 14);
legend (GrSigmaBtName, 'FontSize', 12,  'Location','southwest');
grid on;