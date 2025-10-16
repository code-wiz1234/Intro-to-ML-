%% Ques2PartA_full.m
clc; clear; close all;

mu1 = [0 0]; 
mu2 = [3 3]; 
mu3 = [0 4]; 
mu4 = [4 0];

Sigma1 = [1 0.5; 0.5 1]; 
Sigma2 = [1 -0.3; -0.3 1];
Sigma3 = [1 0; 0 1]; 
Sigma4 = [0.5 0; 0 0.5];

means = {mu1, mu2, mu3, mu4};
covs = {Sigma1, Sigma2, Sigma3, Sigma4};

priors = [0.25 0.25 0.25 0.25];
N = 10000;                        

labels = zeros(N,1);
cum_priors = cumsum(priors);
for i = 1:N
    r = rand;
    if r < cum_priors(1)
        labels(i) = 1;
    elseif r < cum_priors(2)
        labels(i) = 2;
    elseif r < cum_priors(3)
        labels(i) = 3;
    else
        labels(i) = 4;
    end
end

X = zeros(N,2);
for i = 1:N
    lbl = labels(i);
    z = randn(2,1);                % standard normal vector
    L = chol(covs{lbl}, 'lower');  % Cholesky decomposition
    X(i,:) = (means{lbl}' + L*z)'; % sample
end

pred_labels = zeros(N,1);
for i = 1:N
    x = X(i,:)';
    probs = zeros(1,4);
    for j = 1:4
        diff = x - means{j}';
        probs(j) = exp(-0.5 * (diff' / covs{j} * diff)) / sqrt((2*pi)^2 * det(covs{j}));
    end
    [~, pred_labels(i)] = max(probs);
end

accuracy = sum(pred_labels == labels)/N;
fprintf('Classification accuracy (MAP): %.2f%%\n', accuracy*100);

confMat = zeros(4,4);
for j = 1:4
    idx_true = labels == j;        
    for i = 1:4
        confMat(i,j) = sum(pred_labels(idx_true) == i) / sum(idx_true);
    end
end

disp('Empirical Confusion Matrix P(D=i|L=j):');
disp(confMat);

for i = 1:4
    for j = 1:4
        fprintf('P(D=%d|L=%d) = %.3f\t', i, j, confMat(i,j));
    end
    fprintf('\n');
end

figure; hold on;
colors = {'r','g','b','k'};   
markers = {'o','s','^','d'};  

for j = 1:4
    idx = labels == j;         
    scatter(X(idx,1), X(idx,2), 10, colors{j}, markers{j}, 'filled');
end

xlabel('X1'); ylabel('X2');
title('2D Gaussian Samples with True Labels');
legend('Class 1','Class 2','Class 3','Class 4');
grid on; hold off;

%% scatter plot

figure; hold on;


markers = {'.','o','^','s'};  

for j = 1:4
    idx_class = labels == j; 
    
    
    idx_correct = idx_class & (pred_labels == labels);
    idx_incorrect = idx_class & (pred_labels ~= labels);
    
   
    scatter(X(idx_correct,1), X(idx_correct,2), 20, 'g', markers{j}, 'filled');
 
    scatter(X(idx_incorrect,1), X(idx_incorrect,2), 20, 'r', markers{j}, 'filled');
end

xlabel('X1'); ylabel('X2');
title('2D Gaussian Samples: True Class (marker) and Classification (color)');
legend('Class 1 Correct','Class 1 Incorrect','Class 2 Correct','Class 2 Incorrect', ...
       'Class 3 Correct','Class 3 Incorrect','Class 4 Correct','Class 4 Incorrect');
grid on; hold off;
