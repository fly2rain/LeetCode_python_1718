% % online RL without warm-start @ 06/28/2016
clear all;
addpath ('Data_RL_algs\')
%% #1 parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GrAlgOPT = {'fmincon', 'fminconGrad'};
%  run on 07/20/2016 ***********************************
GrSigmaBt = [0, 0.001, 0.02, 0.05]; NSigmaBt = length (GrSigmaBt);
GrT = [50, 100, 150, 200, 250]; NT = length (GrT);
GrMethod = {'noWS', 'noWS', 'WS'}; NMethod = length (GrMethod);
GrTT0 = [2, 20, 1];
endTm   = 5e3; % end   time for long run average measure
starTm  = 1e3; % start time for long run average measure

datIdx=6; T0=42; md.Lo=3; nseRwd=1; nseSt=1; OptAlg=GrAlgOPT{2};
alpha=1;  dirName='v1_'; rngVal=20; valBnd=10; order=1;
FoChoice='itself'; NPeoBC=40; NPeoOL=50; %%%%%%
zetaA=1e-5; zetaC=zetaA; gamma=0;


Beta = [
    0.40,0.25,0.35,0.65,0.10,0.50,0.22,2.00,0.15,0.20,0.32,0.10,0.45,0.00,0.00,0.00,0.00,5;...
    ];

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

% min(GrR(:))
% define the filename for the results, OL Online
rstName = [dirName 'Compare3OL_GrT GrGamma GrNseBt' OptAlg ' datIdx=' num2str(datIdx) ...
    ' T=' num2str(T0) ' NPeoOL=' num2str(NPeoOL) ' alpha=' num2str(alpha) ...
    ' Lo=' num2str(md.Lo) ' NFea=' num2str(md.Lho) ' nseRwd=' num2str(nseRwd) ...
    ' nseSt=' num2str(nseSt) ' zetaC=' num2str(zetaC)  ...
    ' zetaA=' num2str(zetaA) '.mat'];

%% create the matrices to store the results --
GrTheta = zeros( md.Lgo, NPeoOL, NT, NMethod, NSigmaBt );
GrW     = zeros( md.Lho, NPeoOL, NT, NMethod, NSigmaBt);
% (1,1) meanVal, (1,2) stdVal, (2,1) meanPol, (2,2) stdPol
avgPolicy = zeros ( NT, NMethod, NSigmaBt );
stdPolicy = zeros ( NT, NMethod, NSigmaBt );

%% #2 run and test the algorithms
%  id not exists, run for the results
T0_1 = T0 - 1;
if ~exist (rstName, 'file')
    %% generate the data
    for nb = 1 : NSigmaBt
        nseBt = GrSigmaBt (nb);
        [GrBetaBC, GrBetaOL, GrX, GrY, GrA, GrR]  = F_prepareDat_warmStart(...
            'datIdx',datIdx,'TT',T0,'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,...
            'noiseStat',nseSt,'noiseBeta',nseBt,'Beta',Beta,'rngVal',rngVal,...
            'NPeoBC',NPeoBC,'NPeoOL',NPeoOL);
        a0a = GrA (:);
        r0r = GrR (:);
        X0   = reshape ( GrX, [md.Lo, T0_1*NPeoBC] );
        Y0   = reshape ( GrY, [md.Lo, T0_1*NPeoBC] );
        
        [theta0, w0] = F_batchRL( a0a, r0r, X0, Y0, 'gamma', gamma,'maxIter',30, ...
            'mydim',md,'zetaC',zetaC,'zetaA',zetaA, ...
            'OptAlg',OptAlg,'valBnd',valBnd );
        
        rng (rngVal, 'twister');
        GrO0 =  F_simuInitStates (md, NPeoOL, datIdx);
        
        % train the model ---------------------------------------------
        for nm = 1 : NMethod
            nmethod = GrMethod {nm};
            TT0 = GrTT0 (nm);
            for nt = 1 : NT
                T = GrT (nt);
                parfor np = 1 : NPeoOL
                    beta = GrBetaOL (np, :);
                    rng (rngVal, 'twister');
                    if strcmpi( nmethod, 'noWS' )
                        [theta, w] = F_onlineRL (GrO0, np, datIdx, 'gamma',gamma,'TT',T,'theta',[], ...
                            'mydim',md,'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                            'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'TT0',TT0);
                    elseif strcmpi( nmethod, 'WS' )
                        [theta, w] = F_onlineRL_warmSt (GrO0, np, datIdx, a0a, r0r, X0, Y0, ...
                            'gamma',gamma,'TT',T,'theta',theta0,'w',w0, ...
                            'mydim',md,'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                            'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'theta',theta0,'TT0',TT0);
                    end
                    
                    GrTheta(:, np, nt, nm, nb) = theta;
                    GrW(:, np, nt, nm, nb) = w;
                end
            end
        end
        
        % test the results --------------------------------------------
        for nm = 1 : NMethod
            nmethod = GrMethod (nm);
            parfor nt = 1 : NT
                T = GrT (nt);
                
                rng (rngVal, 'twister');
                theta  = GrTheta (:, :, nt, nm, nb);
                thetaN = reshape (theta, [md.Lgo, NPeoOL]);
                NLongRwd = F_tstLongAvgRwd_val_4Each (GrO0,thetaN,[],...
                    'datIdx',datIdx,'EndTime',endTm,'StarTime',starTm,'gamma',gamma,...
                    'alpha', alpha,'mydim',md,'nseRwd',nseRwd,'nseSt',nseRwd, ...
                    'GrBeta',GrBetaOL);
                
                avgPolicy (nt, nm, nb) = mean(NLongRwd);
                stdPolicy (nt, nm, nb) = std (NLongRwd) ./ sqrt(NPeoOL) * 2;
            end
        end
    end
    save ( rstName, 'avgPolicy', 'stdPolicy');
else
    load ( rstName );
end

%% #3 organize and show the results --------------------------
cmpItem = 'T';
sevenColors = {'-<b', '-dk', '-*r', '-om', '-^c', '-sg', '->y'};
figure (2)
set(gcf, 'Unit','centimeters');
set(gcf, 'Position',[0,1,22,16]);
xlimRange = [GrT(1)*0.9, GrT(end)*1.02];

nameOfMethod =  {'noWS-T0=1', 'noWS-T0=20', 'WS-T0=1'};
for nb = 1 : NSigmaBt
    subplot (2,2,nb);
    nseBt = GrSigmaBt (nb);
    
    for nm = 1 : NMethod
        iAvgRwd = avgPolicy (:, nm, nb);
        iStdRwd = stdPolicy (:, nm, nb); % / sqrt(NPeoTs);
        %     plot (GrAlpha, iAvgRwd, sevenColors{ng}, ...
        %         'LineWidth', 2, 'MarkerSize', 6);
        errorbar (GrT, iAvgRwd, iStdRwd, sevenColors{nm}, ...
            'LineWidth', 2, 'MarkerSize', 6);
        hold on;
    end
    xlim ( xlimRange );
    ylim ( [6.25, 9.65] );
    set(0,'DefaultTextFontname', 'Times New Roman');
    xlabel (cmpItem, 'FontSize', 12);
    ylabel ('Expect. Long run average reward', 'FontSize', 10)
    title (['Perform. vs. T, when nseBt=' num2str( nseBt )], 'FontSize', 12);
    legend (nameOfMethod, 'FontSize', 10,  'Location','northwest');
    grid on;
    
end
