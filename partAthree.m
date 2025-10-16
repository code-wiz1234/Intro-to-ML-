%% Part A3

clc; clear; close all;

N = 10000;              
P0 = 0.65;              
P1 = 0.35;              

m0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];

m1 = [1; 1; 1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];

u = rand(1,N) >= P0;       
N0 = sum(u==0);             
N1 = sum(u==1);             
labels = [zeros(N0,1); ones(N1,1)];

L0 = chol(C0,'lower');     
L1 = chol(C1,'lower');     

r0 = repmat(m0,1,N0) + L0*randn(3,N0);  
r1 = repmat(m1,1,N1) + L1*randn(3,N1);  

r0 = r0';   
r1 = r1';

X = [r0; r1]; 
figure;
plot3(r0(:,1), r0(:,2), r0(:,3), '.b'); hold on;
plot3(r1(:,1), r1(:,2), r1(:,3), '.r');
xlabel('x1'); ylabel('x2'); zlabel('x3');
title('3D Samples: Class 0 (Blue) & Class 1 (Red)');
axis equal; grid on;

d = 3; 
invC0 = inv(C0); detC0 = det(C0);
invC1 = inv(C1); detC1 = det(C1);

p0 = zeros(N,1);
p1 = zeros(N,1);

for i = 1:N
    x = X(i,:)';
    p0(i) = (1/((2*pi)^(d/2)*sqrt(detC0))) * exp(-0.5*(x-m0)'*invC0*(x-m0));
    p1(i) = (1/((2*pi)^(d/2)*sqrt(detC1))) * exp(-0.5*(x-m1)'*invC1*(x-m1));
end

ratio = p1 ./ p0;
gamma = P0/P1;            
D = ratio > gamma;        
TP = sum(D==1 & labels==1);  
FP = sum(D==1 & labels==0);  
TN = sum(D==0 & labels==0);  
FN = sum(D==0 & labels==1);  

TPR = TP / N1;               
FPR = FP / N0;               
P_error = FP/N*N0 + FN/N*N1;


fprintf('Threshold gamma: %.3f\n', gamma);
fprintf('TPR: %.3f, FPR: %.3f\n', TPR, FPR);
fprintf('Estimated probability of error: %.3f\n', P_error);


gamma_values = logspace(-2, 3, 100); 
N = size(X,1);        
N0 = sum(labels==0); 
N1 = sum(labels==1); 

TPR = zeros(length(gamma_values),1);
FPR = zeros(length(gamma_values),1);


for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    
   
    D = ratio > gamma;   
    
    TP = sum(D==1 & labels==1);
   
    FP = sum(D==1 & labels==0);
    
    TPR(i) = TP / N1;
    FPR(i) = FP / N0;
end


figure;
plot(FPR, TPR, '-b', 'LineWidth', 2);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve: Likelihood Ratio Classifier');
grid on;
axis([0 1 0 1]);

P_error = zeros(length(gamma_values),1);

for i = 1:length(gamma_values)
    D = ratio > gamma_values(i);
    FP = sum(D==1 & labels==0);
    FN = sum(D==0 & labels==1);
    
    P_error(i) = (FP/N0)*P0 + (FN/N1)*P1; 
end

[minP, idx] = min(P_error); 
gamma_min = gamma_values(idx);  
D_min = ratio > gamma_min;

hold on;
plot(FPR(idx), TPR(idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
legend('ROC Curve', 'Min P(error)');
fprintf('Minimum probability of error: %.4f at gamma = %.3f\n', minP, gamma_min);


P0 = 0.65; 
P1 = 0.35;  
P_error = zeros(length(gamma_values),1);

for i = 1:length(gamma_values)
   
    P_error(i) = FPR(i)*P0 + (1-TPR(i))*P1;
end

[minP, idx] = min(P_error);
gamma_min = gamma_values(idx);
TPR_min = TPR(idx);
FPR_min = FPR(idx);

fprintf('Minimum probability of error = %.4f\n', minP);
fprintf('Gamma (empirical) that achieves min error = %.3f\n', gamma_min);


gamma_theoretical = P0 / P1;
fprintf('Theoretical gamma (from priors) = %.3f\n', gamma_theoretical);

figure;
plot(FPR, TPR, '-b', 'LineWidth', 2); hold on;
plot(FPR_min, TPR_min, 'ro', 'MarkerSize', 8, 'LineWidth', 2); % red circle for min error
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title('ROC Curve with Minimum Error Point');
grid on;
legend('ROC Curve', 'Min P(error)');
