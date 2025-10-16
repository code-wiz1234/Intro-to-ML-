

clear; clc; rng(1); 
N = 10000;           
pL0 = 0.6;          
pL1 = 0.4;           


m0 = [1; 1];
m1 = [4; 4];
C0 = [1 0.5; 0.5 1];
C1 = [1 -0.5; -0.5 1];

N0 = round(N * pL0);
N1 = N - N0;

r0 = randn(N0, length(m0)) * chol(C0) + repmat(m0', N0, 1);
r1 = randn(N1, length(m1)) * chol(C1) + repmat(m1', N1, 1);

X = [r0; r1];         
L = [zeros(N0,1); ones(N1,1)];

m0_hat = mean(r0)';    
m1_hat = mean(r1)';      
C0_hat = cov(r0);       
C1_hat = cov(r1);        

Sw = C0_hat + C1_hat;   
meanDiff = (m1_hat - m0_hat);
Sb = meanDiff * meanDiff';


wLDA = Sw \ meanDiff;    
wLDA = wLDA / norm(wLDA); 

y = X * wLDA;            


tauVals = linspace(min(y), max(y), 200);
TPR = zeros(size(tauVals));
FPR = zeros(size(tauVals));
Perror = zeros(size(tauVals));

for i = 1:length(tauVals)
    tau = tauVals(i);
    D = (y > tau);  
    
    TP = sum((D==1) & (L==1));  
    FP = sum((D==1) & (L==0));   
    TN = sum((D==0) & (L==0));  
    FN = sum((D==0) & (L==1));   

   
    TPR(i) = TP / (TP + FN);   
    FPR(i) = FP / (FP + TN);    

  
    Perror(i) = FPR(i)*pL0 + (1 - TPR(i))*pL1;
end


[minErr, idx] = min(Perror);
bestTau = tauVals(idx);

fprintf('Fisher LDA Classifier:\n');
fprintf('Minimum error = %.4f at threshold = %.3f\n', minErr, bestTau);

figure;
plot(FPR, TPR, 'm-', 'LineWidth', 2); hold on;
plot(FPR(idx), TPR(idx), 'ko', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Fisher LDA Classifier');
grid on;
