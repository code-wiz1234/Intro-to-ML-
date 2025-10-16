clear all; close all; clc;

N = 10000; 
p0 = 0.65; 
p1 = 0.35;


u = rand(1,N) >= p0; 
N0 = length(find(u==0)); 
N1 = length(find(u==1));


m0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
r0 = my_mvnrnd(m0, C0, N0);

m1 = [1; 1; 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];
r1 = my_mvnrnd(m1, C1, N1);


X = [r0; r1];  
L = [zeros(N0,1); ones(N1,1)];  

I = eye(3); 

pdf_class0 = @(x) my_mvnpdf(x, m0', I); 
pdf_class1 = @(x) my_mvnpdf(x, m1', I);   

gamma_values = logspace(-3,3,100); 
TPR = zeros(size(gamma_values));   
FPR = zeros(size(gamma_values));  
Perror = zeros(size(gamma_values));

for gi = 1:length(gamma_values)
    gamma = gamma_values(gi);
    

    px1 = pdf_class1(X);
    px0 = pdf_class0(X);
    ratio = px1 ./ px0;
    
    D = (ratio > gamma);  
    
    TP = sum((D==1) & (L==1));
    FP = sum((D==1) & (L==0));
    FN = sum((D==0) & (L==1));
    TN = sum((D==0) & (L==0));
    
    TPR(gi) = TP / (TP+FN);  
    FPR(gi) = FP / (FP+TN);   
    
    
    Perror(gi) = (FP/(FP+TN))*p0 + (FN/(TP+FN))*p1;
end

figure;
plot(FPR, TPR, 'b-o', 'LineWidth', 1.5); grid on;
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve - Naive Bayes Assumption');

[minPerr, idx] = min(Perror);
bestFPR = FPR(idx);
bestTPR = TPR(idx);

hold on;
plot(bestFPR, bestTPR, 'rs', 'MarkerSize', 10, 'LineWidth', 2);
legend('ROC curve','Min P(error) point');

fprintf('Minimum probability of error (Naive Bayes) = %.4f\n', minPerr);

function X = my_mvnrnd(mu, Sigma, N)
    d = length(mu);
    A = chol(Sigma, 'lower');        
    Z = randn(N, d);                 
    X = Z * A' + repmat(mu(:)', N, 1); 
end

function y = my_mvnpdf(X, mu, Sigma)
    d = size(X,2);
    denom = ((2*pi)^(d/2)) * sqrt(det(Sigma));
    diffs = X - mu;
    exponent = -0.5 * sum((diffs / Sigma) .* diffs, 2);
    y = exp(exponent) / denom;
end
