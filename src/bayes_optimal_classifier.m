%{
                    
PART A: ERM CLASSIFICATION USING THE KNOWLEDGE OF TRUE DATA PDF
%}

clear all, close all,

fprintf('\nPART A: ERM CLASSIFIER \n');

% Data generation parameters
n = 3;
N = 10000; 
mu(:,1) = [-1/2; -1/2; -1/2]; 
Sigma (:,:,1) = [1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
mu(:,2) = [1;1;1]; 
Sigma (:,:,2) = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];
p = [0.65,0.35]; % class priors for labels 0 and 1 respectively

% Data generation
label = rand(1,N) >= p(1); 
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N);

for l = 0:1  % Draw samples from each class pdf
       x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
 
% Display the data generated
figure(1), clf,
plot3(x(1,label==0), x(2,label==0), x(3,label==0), '.b'); hold on;
plot3(x(1,label==1), x(2,label==1), x(3,label==1), '.r'); axis equal;
legend('Class 0','Class 1'), 
title('Data generated from Question 1 and their true labels'),
xlabel('x_1'), ylabel('x_2'), zlabel('x_3')
grid on;

% Calculate theoretical gamma
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
fprintf('Theoretical gamma: %.4f\n', gamma);

% Classification at theoretical gamma 
discriminantScore = log(evalGaussian(x, mu(:,2), Sigma(:,:,2))) - ...
    log(evalGaussian(x, mu(:,1), Sigma(:,:,1)));
decision_theoretical = (discriminantScore >= log(gamma)); 

% Confusion matrix at theoretical gamma 
ind00 = find(decision_theoretical==0 & label==0); 
p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision_theoretical==1 & label==0); 
p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision_theoretical==0 & label==1); 
p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision_theoretical==1 & label==1); 
p11 = length(ind11)/Nc(2); % probability of true positive

% Visualization
% class 0 circle, class 1 +, correct green, incorrect red
figure(2), clf,  
plot3(x(1,ind00),x(2,ind00),x(3,ind00),'og'); hold on,
plot3(x(1,ind10),x(2,ind10),x(3,ind10),'or'); hold on,
plot3(x(1,ind01),x(2,ind01),x(3,ind01),'+r'); hold on,
plot3(x(1,ind11),x(2,ind11),x(3,ind11),'+g'); hold on,
axis equal, grid on,
xlabel('x_1'), ylabel('x_2'), zlabel('x_3');
title('Part A: Bayes Optimal Classification Results at Theoretical Gamma');
legend('TN (Correct)', 'FP (Wrong)', 'FN (Wrong)', 'TP (Correct)');

% Probability of error at theoretical gamma
Perror_theoretical = p10*p(1) + p01*p(2); 
fprintf('P(error) at theoretical gamma: %.4f\n', Perror_theoretical); 
fprintf('TPR = %.4f, FPR = %.4f\n', p11, p10);
fprintf('\n');

% Generate ROC Curve 
[Pfp, Ptp, Pfn, Perror, thresholdList] = ROCcurve(discriminantScore, label, p);

% Empirically optimal gamma 
[min_Perror, min_idx] = min(Perror);
empirical_gamma = exp(thresholdList(min_idx));
fprintf('Empirically optimal gamma: %.4f\n', empirical_gamma);
fprintf('P(error) at empirical gamma: %.4f\n', min_Perror);
fprintf('\n');


% Comparing theoretical and empirically optimal results
fprintf('COMPARISON OF THEORETICAL VS EMPIRICALLY OPTIMAL GAMMA AND PERROR\n');
fprintf('Theoretical gamma: %.4f | Empirical gamma: %.4f\n', gamma, empirical_gamma);
fprintf('Absolute difference:   %.4f\n', abs(gamma - empirical_gamma));
fprintf('Percentage difference:   %.2f%%\n', 100*abs(gamma - empirical_gamma)/gamma);
fprintf('\n');
fprintf('P(error) at theoretical gamma: %.4f | P(error) at empirical gamma: %.4f\n',...
    Perror_theoretical, min_Perror);
fprintf('Difference in P(error):        %.4f\n', abs(Perror_theoretical - min_Perror));
fprintf('\n');

% Plot ROC Curve 
figure(3), clf;
plot(Pfp, Ptp, 'b-', 'LineWidth', 2); hold on;
plot(Pfp(min_idx), Ptp(min_idx), 'go', ...
    'MarkerSize', 10, 'LineWidth', 3); % Marking the empirically optimal gamma point.
[~, theory_idx] = min(abs(thresholdList - log(gamma)));
plot(Pfp(theory_idx), Ptp(theory_idx), 'rs', ...
    'MarkerSize', 10, 'LineWidth', 2); % Marking the theoretical gamma point.
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve for Bayes Optimal Classifier');
legend('ROC Curve', sprintf('Empirical  \\gamma* = %.3f | min P(error) = %.4f',...
    empirical_gamma, min_Perror), sprintf(['Theoretical \\gamma = %.3f | P(error)' ...
    '     = %.4f'], gamma, Perror_theoretical),'Location','southeast');
grid on;

%{
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [Pfp,Ptp,Pfn,Perror,thresholdList] = ROCcurve(discriminantScores,label, p)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Pfp(i) = length(find(decisions==1 & label==0))/length(find(label==0));
    Ptp(i) = length(find(decisions==1 & label==1))/length(find(label==1));
    Pfn(i) = 1 - Ptp(i);
    Perror(i) = p(1)*Pfp(i) + p(2)*Pfn(i);
end
end
%}

%{
PART B: ERM CLASSIFICATION ATTEMPT USING INCORRECT KNOWLEDGE OF DATA
DISTRIBUTION (NAIVE BAYES CLASSIFIER)
%}

fprintf('\nPART B: NAIVE BAYES CLASSIFIER \n');

% Define the identity covariance matrix
Sigma_NB = eye(n); 

% Classification at theoretical gamma (NB)
discriminantScore_NB = log(evalGaussian(x, mu(:,2), Sigma_NB)) - ...
                       log(evalGaussian(x, mu(:,1), Sigma_NB));
decision_NB_theoretical = (discriminantScore_NB >= log(gamma));

% Confusion matrix at theoretical gamma (NB)
ind00_NB = find(decision_NB_theoretical==0 & label==0); 
p00_NB = length(ind00_NB)/Nc(1); % probability of true negative
ind10_NB = find(decision_NB_theoretical==1 & label==0); 
p10_NB = length(ind10_NB)/Nc(1); % probability of false positive
ind01_NB = find(decision_NB_theoretical==0 & label==1); 
p01_NB = length(ind01_NB)/Nc(2); % probability of false negative
ind11_NB = find(decision_NB_theoretical==1 & label==1); 
p11_NB = length(ind11_NB)/Nc(2); % probability of true positive

% Visualization
% class 0 circle, class 1 +, correct green, incorrect red
figure(4), clf,
plot3(x(1,ind00_NB),x(2,ind00_NB),x(3,ind00_NB),'og'); hold on,
plot3(x(1,ind10_NB),x(2,ind10_NB),x(3,ind10_NB),'or'); hold on,
plot3(x(1,ind01_NB),x(2,ind01_NB),x(3,ind01_NB),'+r'); hold on,
plot3(x(1,ind11_NB),x(2,ind11_NB),x(3,ind11_NB),'+g'); hold on,
axis equal, grid on,
xlabel('x_1'), ylabel('x_2'), zlabel('x_3');
title('Part B: Naive Bayes Classification Results at Theoretical Gamma');
legend('TN (Correct)', 'FP (Wrong)', 'FN (Wrong)', 'TP (Correct)');

% Probability of error at theoretical gamma (NB)
Perror_theoretical_NB = p10_NB*p(1) + p01_NB*p(2);
fprintf('Theoretical gamma: %.4f\n', gamma);
fprintf('P(error) at theoretical gamma (NB): %.4f\n', Perror_theoretical_NB);

% Generate ROC curve (NB) 
[Pfp_NB, Ptp_NB, Pfn_NB, Perror_NB, thresholdList_NB] = ...
    ROCcurve(discriminantScore_NB, label, p);

% Empirically optimal gamma (NB) 
[min_Perror_NB, min_idx_NB] = min(Perror_NB);
empirical_gamma_NB = exp(thresholdList_NB(min_idx_NB));
fprintf('\nEmpirically optimal gamma (NB): %.4f\n', empirical_gamma_NB);
fprintf('P(error) at empirical gamma (NB): %.4f\n', min_Perror_NB);
fprintf('\n');

% Comparing theoretical gamma and empirical gamma for Naive Bayes
fprintf('COMPARISON OF NAIVE BAYES THEORETICAL VS EMPIRICALLY OPTIMAL GAMMA AND PERROR\n');
fprintf('Theoretical gamma: %.4f | Empirical gamma (NB): %.4f\n', gamma, empirical_gamma_NB);
fprintf('Absolute difference (NB):   %.4f\n', abs(gamma - empirical_gamma_NB));
fprintf('Percentage difference (NB):   %.2f%%\n', 100*abs(gamma - empirical_gamma_NB)/gamma);
fprintf('\n');
fprintf('P(error) at theoretical gamma (NB): %.4f | P(error) at empirical gamma (NB): %.4f\n',...
    Perror_theoretical_NB, min_Perror_NB);
fprintf('Difference in P(error) (NB):        %.4f\n', abs(Perror_theoretical_NB - min_Perror_NB));
fprintf('\n');

% Plot ROC Curve for Naive Bayes 
figure(5), clf;
plot(Pfp_NB, Ptp_NB, 'b-', 'LineWidth', 2); hold on;
plot(Pfp_NB(min_idx_NB), Ptp_NB(min_idx_NB), 'go',...
    'MarkerSize', 10, 'LineWidth', 3); % Marking the empirically optimal gamma point 
[~, theory_idx_NB] = min(abs(thresholdList_NB - log(gamma)));
plot(Pfp_NB(theory_idx_NB), Ptp_NB(theory_idx_NB), 'rs',...
    'MarkerSize', 10, 'LineWidth', 2);  % Marking the theortical gamma point
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('Part B: ROC Curve for Naive Bayes Classifier');
legend('ROC Curve', sprintf('Empirical  \\gamma* = %.3f | min P(error) = %.4f',...
    empirical_gamma_NB, min_Perror_NB), sprintf(['Theoretical \\gamma = %.3f | P(error)' ...
    '     = %.4f'], gamma, Perror_theoretical_NB),'Location','southeast');
grid on;

% Comparison of True Bayes vs Naive Bayes
% Compare ROC curves of from Part A and B 
figure(6), clf;
plot(Pfp, Ptp, 'b-', 'LineWidth', 2.5); hold on;
plot(Pfp_NB, Ptp_NB, 'r-', 'LineWidth', 2.5);

% Mark minimum P(error) points for both
plot(Pfp(min_idx), Ptp(min_idx), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
plot(Pfp_NB(min_idx_NB), Ptp_NB(min_idx_NB), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Comparison: Impact of Model Mismatch');
legend(sprintf('Part A: Bayes Optimal (Min P(error) = %.4f)', min_Perror), ...
       sprintf('Part B: Naive Bayes (Min P(error) = %.4f)', min_Perror_NB), ...
       'Part A Min Error Point', 'Part B Min Error Point', 'Location', 'southeast');
grid on;

% Comparing performance of the Bayes Optimal (True Model) vs Naive Bayes (Mismatch Model)
fprintf('\nComparison of Bayes Optimal (True Model) vs Naive Bayes (Mismatch Model)\n');
fprintf('Part A (Bayes Optimal) - Min P(error): %.4f\n', min_Perror);
fprintf('Part B (Naive Bayes) - Min P(error): %.4f\n', min_Perror_NB);
fprintf('Performance Degradation: %.4f (%.2f%% worse)\n', ...
        min_Perror_NB - min_Perror, ...
        100*(min_Perror_NB - min_Perror)/min_Perror);


%PART C: FISHER LDA CLASSIFIER

fprintf('\nPART C: FISHER LDA CLASSIFIER \n');

% Separate data by class
x0 = x(:, label==0);  % Class 0 samples
x1 = x(:, label==1);  % Class 1 samples

% Estimate mean vectors and covariance matrices from samples 
mu0hat = mean(x0, 2);
S0hat = cov(x0');
mu1hat = mean(x1, 2);
S1hat = cov(x1');

% Compute between-class and within-class scatter matrices 
Sb = (mu0hat - mu1hat) * (mu0hat - mu1hat)';  % Between-class scatter
Sw = S0hat + S1hat;                             % Within-class scatter

% Find Fisher LDA projection vector 
[V, D] = eig(inv(Sw) * Sb);
[~, ind] = sort(diag(D), 'descend');
wLDA = V(:, ind(1));  % Fisher LDA projection vector
yLDA = wLDA' * x; % Project all data onto the Fisher LDA direction

% Ensure class 1 projects to positive side 
if mean(yLDA(label==1)) < mean(yLDA(label==0))
    wLDA = -wLDA;
    yLDA = -yLDA;
end

% Classification at theoretical gamma using LDA projections 
discriminantScore_LDA = yLDA;

% Generate ROC curve (LDA) 
[Pfp_LDA, Ptp_LDA, Pfn_LDA, Perror_LDA, thresholdList_LDA] = ...
    ROCcurve(discriminantScore_LDA, label, p);

% Find empirically optimal gamma for LDA
[min_Perror_LDA, min_idx_LDA] = min(Perror_LDA);
empirical_gamma_LDA = thresholdList_LDA(min_idx_LDA);
fprintf('Empirically optimal gamma (LDA): %.4f\n', empirical_gamma_LDA);
fprintf('P(error) at empirical gamma (LDA): %.4f\n', min_Perror_LDA);

% Visualize LDA projections
figure(7), clf;
subplot(2,1,1), 
plot3(x(1,label==0), x(2,label==0), x(3,label==0), 'bo'); hold on;
plot3(x(1,label==1), x(2,label==1), x(3,label==1), 'r+');
axis equal, grid on;
xlabel('x_1'), ylabel('x_2'), zlabel('x_3');
title('Part C: Original 3D Data');
legend('Class 0', 'Class 1');

subplot(2,1,2),
plot(yLDA(label==0), zeros(1,Nc(1)), 'bo'); hold on;
plot(yLDA(label==1), zeros(1,Nc(2)), 'r+');
axis equal, grid on;
xlabel('LDA Projection'), ylabel('');
title('Part C: Data Projected onto Fisher LDA Direction');
legend('Class 0', 'Class 1');

% Plot ROC Curve for Fisher LDA
figure(8), clf;
plot(Pfp_LDA, Ptp_LDA, 'b-', 'LineWidth', 2); hold on;
plot(Pfp_LDA(min_idx_LDA), Ptp_LDA(min_idx_LDA), 'go',...
    'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('Part C: ROC Curve for Fisher LDA Classifier');
legend('ROC Curve', sprintf('Empirical  \\gamma* = %.3f | min P(error) = %.4f',...
    empirical_gamma_LDA, min_Perror_LDA), 'Location', 'southeast');
grid on;

% Final Comparison: Parts A, B, and C
figure(9), clf;
plot(Pfp, Ptp, 'b-', 'LineWidth', 2.5); hold on;
plot(Pfp_NB, Ptp_NB, 'r-', 'LineWidth', 2.5);
plot(Pfp_LDA, Ptp_LDA, 'c-.', 'LineWidth', 2.5);
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Comparison: All Three Classifiers');
legend(sprintf('Part A: True Model (%.4f)', min_Perror), ...
       sprintf('Part B: Naive Bayes (%.4f)', min_Perror_NB), ...
       sprintf('Part C: Fisher LDA (%.4f)', min_Perror_LDA), ...
       'Location', 'southeast');
grid on;

% Print final comparison
fprintf('\n=== FINAL COMPARISON ===\n');
fprintf('Part A (True Model)  - Min P(error): %.4f\n', min_Perror);
fprintf('Part B (Naive Bayes) - Min P(error): %.4f\n', min_Perror_NB);
fprintf('Part C (Fisher LDA)  - Min P(error): %.4f\n', min_Perror_LDA);

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [Pfp,Ptp,Pfn,Perror,thresholdList] = ROCcurve(discriminantScores,label, p)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2, max(sortedScores)+eps];
for i = 1:length(thresholdList)
    tau = thresholdList(i);
    decisions = (discriminantScores >= tau);
    Pfp(i) = length(find(decisions==1 & label==0))/length(find(label==0));
    Ptp(i) = length(find(decisions==1 & label==1))/length(find(label==1));
    Pfn(i) = 1 - Ptp(i);
    Perror(i) = p(1)*Pfp(i) + p(2)*Pfn(i);
end
end

