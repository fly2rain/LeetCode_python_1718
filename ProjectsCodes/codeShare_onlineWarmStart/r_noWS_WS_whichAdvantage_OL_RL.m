% % online RL without warm-start @ 06/28/2016
clear all;
addpath ('Data_RL_algs\')
%% #1 parameter setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GrAlgOPT = {'fmincon', 'fminconGrad'};
endTm    = 5e3; % end   time for long run average measure
starTm   = 1e3; % start time for long run average measure

% % verion 1 @ 07/25/2016 important -----------------------------------------
%  the maximum trajectory length is 250
% GrGamma   = [0, 0.3, 0.6, 0.9, 0.95, 1];  NGamma=length(GrGamma);
% GrSigmaBt = [0, 0.001, 0.01, 0.05]; NSigmaBt = length (GrSigmaBt);
% GrT = [50, 100, 150, 200, 250]; NT = length (GrT);
% GrMethod = {'noWS', 'WS#1', 'WS#2', 'WS#1&2'}; NMethod = length (GrMethod);
% 
% datIdx=6; T0=42; md.Lo=3; nseRwd=1; nseSt=1; OptAlg=GrAlgOPT{2};
% alpha=1;  dirName='v3_'; rngVal=20; valBnd=10; zetaA=1e-5; zetaC=zetaA; order=1;
% FoChoice='itself'; NPeoBC=40; NPeoOL=50; %%%%%%

% verion 2 @ 08/01/2016 longer trajectory is involved, T=500 -----------------
GrGamma   = [0, 0.3, 0.6, 0.9, 0.95, 1];  NGamma=length(GrGamma);
GrSigmaBt = [0, 0.001, 0.01, 0.05]; NSigmaBt = length (GrSigmaBt);
GrT = linspace (50, 500, 6); NT = length (GrT);
GrMethod = {'noWS', 'WS#1', 'WS#2', 'WS#1&2'}; NMethod = length (GrMethod);

datIdx=6; T0=42; md.Lo=3; nseRwd=1; nseSt=1; OptAlg=GrAlgOPT{2};
alpha=1;  dirName='v3_'; rngVal=20; valBnd=10; zetaA=1e-5; zetaC=zetaA; order=1;
FoChoice='itself'; NPeoBC=40; NPeoOL=40; %%%%%%

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

% define the filename for the results, OL Online
rstName = [dirName 'noWS_WS_whichAdvantage_GrT GrGamma GrNseBt' OptAlg ' datIdx=' num2str(datIdx) ...
    ' T=' num2str(T0) ' NPeoOL=' num2str(NPeoOL) ' alpha=' num2str(alpha) ...
    ' Lo=' num2str(md.Lo) ' NFea=' num2str(md.Lho) ' nseRwd=' num2str(nseRwd) ...
    ' nseSt=' num2str(nseSt) ' zetaC=' num2str(zetaC)  ...
    ' zetaA=' num2str(zetaA) ' rngVal' num2str(rngVal) '.mat'];

%% create the matrices to store the results --
GrTheta   = zeros( md.Lgo, NPeoOL, NT, NMethod, NGamma, NSigmaBt);
% GrW     = zeros( md.Lho, NPeoOL, NT, NMethod, NGamma, NSigmaBt);
% (1,1) meanVal, (1,2) stdVal, (2,1) meanPol, (2,2) stdPol
avgPolicy = zeros ( NT, NMethod, NGamma, NSigmaBt );
stdPolicy = zeros ( NT, NMethod, NGamma, NSigmaBt );

T0_1    = T0 - 1;
%% #2 run and test the algorithms
%  id not exists, run for the results
if ~exist (rstName, 'file')
    for nb = 1 : NSigmaBt
        nseBt = GrSigmaBt (nb);
        %% generate the data
        [GrBetaBC, GrBetaOL, GrX, GrY, GrA, GrR]  = F_prepareDat_warmStart (...
            'datIdx',datIdx,'TT',T0,'alpha',alpha,'mydim',md,'noiseRwd',nseRwd,...
            'noiseStat',nseSt,'noiseBeta',nseBt,'Beta',Beta,'rngVal',rngVal,...
            'NPeoBC',NPeoBC,'NPeoOL',NPeoOL);
        a0a = GrA (:);
        r0r = GrR (:);
        X0  = reshape ( GrX, [md.Lo, T0_1*NPeoBC] );
        Y0  = reshape ( GrY, [md.Lo, T0_1*NPeoBC] );
        %         rng (rngVal, 'twister');
        %         [theta_0, w_0] = F_batchRL (a0a, r0r, X0, Y0, 'gamma',0,'maxIter',30, ...
        %             'mydim',md,'zetaC',zetaC,'zetaA',zetaA, ...
        %             'OptAlg',OptAlg,'valBnd',valBnd );
        
        for ng = 1 : NGamma
            rng (rngVal, 'twister');
            GrO0 = F_simuInitStates (md, NPeoOL, datIdx);
            gamma = GrGamma (ng);
            [theta_r, w_r] = F_batchRL (a0a, r0r, X0, Y0, 'gamma',gamma,'maxIter',30, ...
                'mydim',md,'zetaC',zetaC,'zetaA',zetaA, ...
                'OptAlg',OptAlg,'valBnd',valBnd );
            
            % train the model *************************************
            for nm = 1 : NMethod
                nmethod = GrMethod {nm};
                
                for nt = 1 : NT
                    T = GrT (nt);
                    parfor np = 1 : NPeoOL
                        rng (rngVal, 'twister');
                        beta = GrBetaOL (np, :);
                        if strcmpi( nmethod, GrMethod{1} )
                            theta = F_onlineRL (GrO0, np, datIdx, 'gamma',gamma,'TT',T,'theta',[], ...
                                'mydim',md,'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                                'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'TT0',20);
                        elseif strcmpi( nmethod, GrMethod{2} )
                            theta = F_onlineRL (GrO0, np, datIdx, ...
                                'gamma',gamma,'TT',T,'theta',theta_r,'w',w_r,'mydim',md,...
                                'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                                'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'TT0',1);
                        elseif strcmpi( nmethod, GrMethod{3} )
                            theta = F_onlineRL_warmSt (GrO0, np, datIdx, a0a, r0r, X0, Y0, ...
                                'gamma',gamma,'TT',T,'theta',[],'w',[],'mydim',md,...
                                'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                                'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'TT0',1);
                        elseif strcmpi( nmethod, GrMethod{4} )
                            theta = F_onlineRL_warmSt (GrO0, np, datIdx, a0a, r0r, X0, Y0, ...
                                'gamma',gamma,'TT',T,'theta',theta_r,'w',w_r,'mydim',md,...
                                'zetaC',zetaC,'zetaA',zetaA,'OptAlg',OptAlg,'valBnd',valBnd, ...
                                'beta',beta,'nseRwd',nseRwd,'nseSt',nseSt,'TT0',1);
                        end
                        GrTheta(:, np, nt, nm, ng, nb) = theta;
                        % GrW(:, np, nt, nm, ng, nb) = w;
                    end
                end
                
                % test the results ************************************
                parfor nt = 1 : NT
                    rng (rngVal, 'twister');
                    thetaTmp  = GrTheta(:, :, nt, nm, ng, nb);
                    thetaN = reshape (thetaTmp, [md.Lgo, NPeoOL]);
                    NLongRwd = F_tstLongAvgRwd_val_4Each (GrO0,thetaN,[],...
                        'datIdx',datIdx,'EndTime',endTm,'StarTime',starTm,'gamma',gamma,...
                        'alpha', alpha,'mydim',md,'nseRwd',nseRwd,'nseSt',nseRwd, ...
                        'GrBeta',GrBetaOL);
                    
                    avgPolicy (nt, nm, ng, nb) = mean(NLongRwd);
                    stdPolicy (nt, nm, ng, nb) = std (NLongRwd) ./ sqrt(NPeoOL) * 2;
                end
                
            end
        end
    end
    save ( rstName, 'avgPolicy', 'stdPolicy');
else
    load ( rstName );
end

%% #3 organize the results --------------------------
cmpItem = 'T';
sevenColors = { '-<b', '-dr', '-*k', '-^c', '-om', '-sg', '->y'};
xlimRange = [GrT(1)*0.9, GrT(end)*1.02];

for nb = 1 : NSigmaBt
    figure (nb)
    set(gcf, 'Unit','centimeters');
    set(gcf, 'Position',[0,1,36,16]);
    
    for ng = 1 : NGamma
        subplot (2,3,ng);
        gamma = GrGamma (ng);
        
        for nm = 2 : NMethod
            iAvgRwd = avgPolicy (:, nm, ng, nb);
            iStdRwd = stdPolicy (:, nm, ng, nb); % / sqrt(NPeoTs);
            errorbar (GrT, iAvgRwd, iStdRwd, sevenColors{nm}, ...
                'LineWidth', 2, 'MarkerSize', 6);
            hold on;
        end
        % set the figure property
        xlim ( xlimRange );
        % ylim auto;
        %         ylim ( [7, 10.3] );
        set(0,'DefaultTextFontname', 'Times New Roman');
        xlabel (cmpItem, 'FontSize', 14);
        ylabel ('Expect. Long run average reward', 'FontSize', 12)
        title (['Perform. vs. T, when \gamma=' num2str( gamma )], 'FontSize', 14);
        legend (GrMethod(2 : NMethod), 'FontSize', 10,  'Location','southeast');
        grid on;
    end
end
