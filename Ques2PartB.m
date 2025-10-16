clc; clear; close all;

N = 10000;
priors = [0.25 0.25 0.25 0.25];
d = 2;

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

labels = zeros(N,1);
cum_priors = cumsum(priors);
X = zeros(N, d);

for n = 1:N
    r = rand();
    lbl = find(r <= cum_priors, 1);
    labels(n) = lbl;

    L = chol(covs{lbl}, 'lower');
    X(n,:) = means{lbl} + randn(1, d) * L;
end

Lambda = [0 10 10 100;
           1  0 10 100;
           1  1  0 100;
           1  1  1   0];

post = zeros(N,4);
for n = 1:N
    x = X(n,:)';
    likelihoods = zeros(1,4);
    for j = 1:4
        diff = x - means{j}';
        likelihoods(j) = exp(-0.5*(diff'/covs{j}*diff)) / sqrt((2*pi)^2 * det(covs{j}));
    end
    post(n,:) = likelihoods / sum(likelihoods);
end

ERM_labels = zeros(N,1);
for n = 1:N
    risks = Lambda * post(n,:)'; 
    [~, ERM_labels(n)] = min(risks);
end

total_risk = 0;
for n = 1:N
    total_risk = total_risk + Lambda(ERM_labels(n), labels(n));
end
min_expected_risk = total_risk / N;

fprintf('Empirical minimum expected risk (ERM): %.4f\n', min_expected_risk);

confMat_ERM = zeros(4,4);
for j = 1:4
    idx_true = labels == j;
    for i = 1:4
        confMat_ERM(i,j) = sum(ERM_labels(idx_true) == i) / sum(idx_true);
    end
end

disp('ERM Confusion Matrix P(D=i | L=j):');
disp(confMat_ERM);

figure; hold on;
colors = {'r','g','b','k'};
markers = {'o','s','^','d'};
for j = 1:4
    idx = labels == j;
    scatter(X(idx,1), X(idx,2), 10, colors{j}, markers{j}, 'filled');
end
title('Generated Gaussian Samples');
xlabel('x_1'); ylabel('x_2');
legend('Class 1','Class 2','Class 3','Class 4');
grid on;
hold off;
