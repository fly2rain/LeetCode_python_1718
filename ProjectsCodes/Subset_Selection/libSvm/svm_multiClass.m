%Input
% train_X   n x d matrix, input data, n --- the number of training samples
%                                     d --- the dimension of data
% train_L   n x 1 matrix, labels of training samples
% test_X    n x d matrix,
% test_L    n x 1 matrix,
%Output
% sp  ������ı��?
% ac  Accuracy
% path_savemodel  �洢ѵ����SVM-model

function [sp, ac, bestc, bestg, smodel] = svm_multiClass(train_X, train_L, test_X, test_L, path_savemodel)

nc = size( unique(train_L), 1);
%% rescaling training data
fprintf('Scaling training data...\n');
% ����[0, 1], ѵ��������Լ�����һ�����, �������������� [-1, 1]
F_max = max(train_X,[],1);   %��ά��������������?
F_min = min(train_X,[],1);   %
strain_X = (train_X - repmat(F_min,size(train_X,1),1))*...
    spdiags(1./( F_max-F_min)',0,size(train_X,2),size(train_X,2));

strain_X = (strain_X - 0.5)*2;  % ��һ������[-1 ,1]

%% parameter selection          cross-validation
fprintf('Cross validation...\n');
bestcv = 0; bestc = 2^(-1); bestg = 2^(-4);
for log2c = 2:10,   %-1:8   1:3
    for log2g = -10:-2,      % -4:2
% for log2c = 10:10,   %-1:8   1:3
%     for log2g = -2:-2,      % -4:2
        cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -n ', num2str(nc)];
        % cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g),'-t', 0, ' -n ', num2str(nc)];  %lhc 2012 09 29
        cv = svmtrain(double(train_L), strain_X, cmd);
        if (cv >= bestcv),
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end

fprintf('Best c=%g, g=%g\n', bestc, bestg);   %�þ�ɲ鿴ѡ��Ĳ���
%% ѵ��ģ��
fprintf('Training...\n');
parameter = ['-c ', num2str(bestc), ' -g ', num2str(bestg), ' -n ', num2str(nc)]; % ע�������仰�����?
smodel  = svmtrain(double(train_L), strain_X, parameter);                 %���ַ����ͣ����ֺͱ�������ת��Ϊ�ַ���У�?
path_smodel =  strcat(path_savemodel, '/smodel.mat');
% save(path_smodel, 'smodel'); % �洢ģ��
%% rescaling testing data
fprintf('Scaling testing data...\n');
stest_X = (test_X - repmat(F_min,size(test_X,1),1))*...
    spdiags(1./(F_max-F_min)',0,size(test_X,2),size(test_X,2));
stest_X = (stest_X - 0.5)*2;  % ��һ������[-1 ,1]

%% ����
fprintf('Testing...\n');
[sp, ac] = svmpredict(double(test_L), stest_X, smodel);