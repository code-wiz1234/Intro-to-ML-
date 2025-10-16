
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

%% erm decision
ERM_labels = zeros(N,1);
for n = 1:N
    risks = Lambda * post(n,:)'; 
    [~, ERM_labels(n)] = min(risks);
end

%% min expected risk
total_risk = 0;
for n = 1:N
    total_risk = total_risk + Lambda(ERM_labels(n), labels(n));
end
min_expected_risk = total_risk / N;

fprintf('Empirical minimum expected risk (ERM): %.3f\n', min_expected_risk);

confMat_ERM = zeros(4,4);
for j = 1:4
    idx_true = labels == j;
    for i = 1:4
        confMat_ERM(i,j) = sum(ERM_labels(idx_true) == i) / sum(idx_true);
    end
end

disp('ERM Confusion Matrix P(D=i|L=j):');
disp(confMat_ERM);
