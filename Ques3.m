%% Minimum Probability-of-Error Classifier (Gaussian Bayes) - MATLAB
% Implements Bayes/ERM classifier from scratch for Wine Quality & HAR datasets

clear; close all; clc;

%% Loading datasets
red  = readtable('C:\Users\Owner\Downloads\winequality-red.csv');
white = readtable('C:\Users\Owner\Downloads\winequality-white.csv');
wineData = [red; white];

X_wine = wineData{:,1:11}; % features
y_wine = wineData.quality; % labels
classes_wine = unique(y_wine);
numClasses_wine = length(classes_wine);

%% Splitting data 70-30
cv = cvpartition(size(X_wine,1),'HoldOut',0.3);
X_train = X_wine(training(cv), :); y_train = y_wine(training(cv));
X_test  = X_wine(test(cv), :);     y_test  = y_wine(test(cv));

%% We are estimating mean, covariance andpriors 
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

%% Here we classify test samples
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

%% This is the confusion matric
confMat = confusionmat(y_test, y_pred);
accuracy = mean(y_pred == y_test);
errorRate = 1 - accuracy;

fprintf('Wine Quality Classifier Accuracy: %.2f%%\n', accuracy*100);
fprintf('Error Rate: %.2f%%\n', errorRate*100);
disp('Confusion Matrix (rows=true, cols=predicted):'); disp(confMat);

figure; confusionchart(confMat); title('Wine Quality Gaussian Classifier');

%% We perform pca visusalization
X_std = (X_wine - mean(X_wine)) ./ std(X_wine);
[coeff, score, ~] = pca(X_std);

figure; gscatter(score(:,1), score(:,2), y_wine);
xlabel('PC1'); ylabel('PC2'); title('Wine Quality PCA 2D');

%% This is the har dataset

x_train_path = 'C:\Users\Owner\Downloads\UCI HAR Dataset\train\X_train.txt';
y_train_path = 'C:\Users\Owner\Downloads\UCI HAR Dataset\train\y_train.txt';
x_test_path  = 'C:\Users\Owner\Downloads\UCI HAR Dataset\test\X_test.txt';
y_test_path  = 'C:\Users\Owner\Downloads\UCI HAR Dataset\test\y_test.txt';

X_train = dlmread(x_train_path); 
y_train = dlmread(y_train_path);
X_test  = dlmread(x_test_path);  
y_test  = dlmread(y_test_path);

classes = unique(y_train); 
numClasses = length(classes);
numFeatures = size(X_train,2);

lambda = 1;  
mu = zeros(numClasses, numFeatures);
Sigma = zeros(numFeatures, numFeatures, numClasses);
priors = zeros(numClasses,1);

for k = 1:numClasses
    Xk = X_train(y_train == classes(k), :);
    mu(k,:) = mean(Xk,1);
    
    Sigma(:,:,k) = diag(var(Xk) + lambda); 
    priors(k) = size(Xk,1)/size(X_train,1);
end


numTest = size(X_test,1);
logPosterior = zeros(numTest, numClasses);

for i = 1:numTest
    x = X_test(i,:);
    for k = 1:numClasses
        
        diff = x - mu(k,:);
        invDiag = 1 ./ diag(Sigma(:,:,k))';
        logDet = sum(log(diag(Sigma(:,:,k))));
        logLikelihood = -0.5 * sum(diff.^2 .* invDiag) - 0.5*logDet - (numFeatures/2)*log(2*pi);
        logPosterior(i,k) = logLikelihood + log(priors(k));
    end
end

[~, idx] = max(logPosterior,[],2);
y_pred = classes(idx);


confMat = confusionmat(y_test, y_pred);
accuracy = mean(y_pred == y_test);
errorRate = 1 - accuracy;

fprintf('HAR Classifier Accuracy: %.2f%%\n', accuracy*100);
disp('HAR Confusion Matrix:'); disp(confMat);

figure; confusionchart(confMat);
title('HAR Gaussian Classifier (Diagonal Covariance)');

X_std = (X_train - mean(X_train)) ./ std(X_train);
[coeff, score, ~] = pca(X_std);

figure; gscatter(score(:,1), score(:,2), y_train);
xlabel('PC1'); ylabel('PC2'); title('HAR PCA Visualization');
