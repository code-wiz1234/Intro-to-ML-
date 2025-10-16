%% Minimum Probability-of-Error Classifier (Gaussian Bayes) - MATLAB
% Implements Bayes/ERM classifier from scratch for Wine Quality & HAR datasets

clear; close all; clc;

red  = readtable('C:\Users\Owner\Downloads\winequality-red.csv');
white = readtable('C:\Users\Owner\Downloads\winequality-white.csv');
wineData = [red; white];

X_wine = wineData{:,1:11}; % features
y_wine = wineData.quality; % labels
classes_wine = unique(y_wine);
numClasses_wine = length(classes_wine);

%% =================== 2. Split Data (70/30) ===================
cv = cvpartition(size(X_wine,1),'HoldOut',0.3);
X_train = X_wine(training(cv), :); y_train = y_wine(training(cv));
X_test  = X_wine(test(cv), :);     y_test  = y_wine(test(cv));

%% =================== 3. Estimate Mean, Covariance, Priors ===================
lambda = 0.01; % regularization
mu = zeros(numClasses_wine, size(X_wine,2));
Sigma = zeros(size(X_wine,2), size(X_wine,2), numClasses_wine);
priors = zeros(numClasses_wine,1);

for k = 1:numClasses_wine
    Xk = X_train(y_train == classes_wine(k), :);
    mu(k,:) = mean(Xk,1);
    Sigma(:,:,k) = cov(Xk) + lambda*eye(size(X_wine,2));
    priors(k) = size(Xk,1)/size(X_train,1);
end

%% =================== 4. Classify Test Samples ===================
numTest = size(X_test,1);
logPosterior = zeros(numTest, numClasses_wine);

for k = 1:numClasses_wine
    invSigma = inv(Sigma(:,:,k));
    detSigma = det(Sigma(:,:,k));
    for i = 1:numTest
        x = X_test(i,:)';
        diff = x - mu(k,:)';
        logLikelihood = -0.5*log(detSigma) -0.5*(diff')*invSigma*diff;
        logPosterior(i,k) = logLikelihood + log(priors(k));
    end
end

[~, idx] = max(logPosterior,[],2);
y_pred = classes_wine(idx);

%% =================== 5. Confusion Matrix & Accuracy ===================
confMat = confusionmat(y_test, y_pred);
accuracy = mean(y_pred == y_test);
errorRate = 1 - accuracy;

fprintf('Wine Quality Classifier Accuracy: %.2f%%\n', accuracy*100);
fprintf('Error Rate: %.2f%%\n', errorRate*100);
disp('Confusion Matrix (rows=true, cols=predicted):'); disp(confMat);

figure; confusionchart(confMat); title('Wine Quality Gaussian Classifier');

%% =================== 6. 3D PCA Visualization ===================
X_std = (X_wine - mean(X_wine)) ./ std(X_wine);
[coeff, score, ~] = pca(X_std);

figure; gscatter(score(:,1), score(:,2), y_wine);
xlabel('PC1'); ylabel('PC2'); title('Wine Quality PCA 2D');

%% =================== 7. HAR Dataset Example ===================
% Paths to UCI HAR dataset
x_train_path = 'C:\Users\Owner\Downloads\UCI HAR Dataset\train\X_train.txt';
y_train_path = 'C:\Users\Owner\Downloads\UCI HAR Dataset\train\y_train.txt';
x_test_path  = 'C:\Users\Owner\Downloads\UCI HAR Dataset\test\X_test.txt';
y_test_path  = 'C:\Users\Owner\Downloads\UCI HAR Dataset\test\y_test.txt';

X_train_HAR = dlmread(x_train_path); y_train_HAR = dlmread(y_train_path);
X_test_HAR  = dlmread(x_test_path);  y_test_HAR  = dlmread(y_test_path);

classes_HAR = unique(y_train_HAR); numClasses_HAR = length(classes_HAR);

% Estimate parameters
mu_HAR = zeros(numClasses_HAR, size(X_train_HAR,2));
Sigma_HAR = zeros(size(X_train_HAR,2), size(X_train_HAR,2), numClasses_HAR);
priors_HAR = zeros(numClasses_HAR,1);

for k = 1:numClasses_HAR
    Xk = X_train_HAR(y_train_HAR == classes_HAR(k), :);
    mu_HAR(k,:) = mean(Xk,1);
    Sigma_HAR(:,:,k) = cov(Xk) + lambda*eye(size(X_train_HAR,2));
    priors_HAR(k) = size(Xk,1)/size(X_train_HAR,1);
end

% Classify
numTest = size(X_test_HAR,1);
logPosterior = zeros(numTest, numClasses_HAR);
for k = 1:numClasses_HAR
    invSigma = inv(Sigma_HAR(:,:,k));
    detSigma = det(Sigma_HAR(:,:,k));
    for i = 1:numTest
        x = X_test_HAR(i,:)';
        diff = x - mu_HAR(k,:)';
        logLikelihood = -0.5*log(detSigma) -0.5*(diff')*invSigma*diff;
        logPosterior(i,k) = logLikelihood + log(priors_HAR(k));
    end
end
[~, idx] = max(logPosterior,[],2);
y_pred_HAR = classes_HAR(idx);

% Confusion Matrix & Accuracy
confMat_HAR = confusionmat(y_test_HAR, y_pred_HAR);
accuracy_HAR = mean(y_pred_HAR == y_test_HAR);
fprintf('HAR Classifier Accuracy: %.2f%%\n', accuracy_HAR*100);
disp('HAR Confusion Matrix:'); disp(confMat_HAR);

figure; confusionchart(confMat_HAR); title('HAR Gaussian Classifier');


