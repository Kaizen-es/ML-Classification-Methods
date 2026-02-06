%{
                        SVM AND MLP CLASSIFICATION
%}

clear all, close all,

% Set random seed for reproducibility
rng(5644, 'twister');

fprintf('SVM and MLP Classification\n\n');

% DATA GENERATION
% Data parameters
r_minus1 = 2;    % Inner circle radius (class -1)
r_plus1 = 4;     % Outer circle radius (class +1)
sigma = 1;      

N_train = 1000;  % Training samples
N_test = 10000;  % Test samples

fprintf('DATA GENERATION\n');
fprintf('Parameters: r_(-1)=%d, r_(+1)=%d, sigma=%d\n', r_minus1, r_plus1, sigma);

% Generate training data
[x_train, labels_train] = generate_concentric_circles(N_train, r_minus1, r_plus1, sigma);
fprintf('Training set: %d samples (Class -1: %d, Class +1: %d)\n', ...
    N_train, sum(labels_train==-1), sum(labels_train==1));

% Generate test data
[x_test, labels_test] = generate_concentric_circles(N_test, r_minus1, r_plus1, sigma);
fprintf('Test set: %d samples (Class -1: %d, Class +1: %d)\n\n', ...
    N_test, sum(labels_test==-1), sum(labels_test==1));

% Visualize training data
figure(1), clf,
plot(x_train(1, labels_train==-1), x_train(2, labels_train==-1), 'bo', 'MarkerSize', 4); hold on,
plot(x_train(1, labels_train==1), x_train(2, labels_train==1), 'r+', 'MarkerSize', 4);
axis equal, grid on,
xlabel('x_1'), ylabel('x_2');
title('Training Data: Concentric Circles');
legend('Class -1 (inner)', 'Class +1 (outer)', 'Location', 'best');

% PART 1: SUPPORT VECTOR MACHINE (GAUSSIAN KERNEL)
fprintf('PART 1: SVM WITH GAUSSIAN KERNEL\n');
fprintf('Using K-fold cross-validation for hyperparameter selection\n\n');

% K-fold cross-validation setup
K = 10;
dummy = ceil(linspace(0, N_train, K+1));
for k = 1:K
    ind_partition_limits(k,:) = [dummy(k)+1, dummy(k+1)];
end

% Hyperparameter grid 
C_list = 10.^linspace(-1, 5, 11);          % Box constraint values (expanded range)
sigma_list = 10.^linspace(-1, 3, 11);      % Kernel scale values (expanded range)

fprintf('Hyperparameter grid:\n');
fprintf('C (BoxConstraint): [%.2e, %.2e] (%d values)\n', ...
    min(C_list), max(C_list), length(C_list));
fprintf('sigma (KernelScale): [%.2e, %.2e] (%d values)\n\n', ...
    min(sigma_list), max(sigma_list), length(sigma_list));

% K-fold cross-validation 
fprintf('%d-fold cross-validation: \n', K);
P_correct = zeros(length(C_list), length(sigma_list));

for sigma_counter = 1:length(sigma_list)
    fprintf('  sigma = %.2e (%d/%d)\n', sigma_list(sigma_counter), ...
        sigma_counter, length(sigma_list));
    sigma_val = sigma_list(sigma_counter);
    
    for C_counter = 1:length(C_list)
        C_val = C_list(C_counter);
        N_correct = zeros(K, 1);
        
        for k = 1:K
            % Partition data 
            ind_validate = ind_partition_limits(k,1):ind_partition_limits(k,2);
            x_validate = x_train(:, ind_validate);
            l_validate = labels_train(ind_validate);
            
            if k == 1
                ind_train_fold = ind_partition_limits(k,2)+1:N_train;
            elseif k == K
                ind_train_fold = 1:ind_partition_limits(k,1)-1;
            else
                ind_train_fold = [1:ind_partition_limits(k,1)-1, ind_partition_limits(k,2)+1:N_train];
            end
            
            x_train_fold = x_train(:, ind_train_fold);
            l_train_fold = labels_train(ind_train_fold);
            
            % Train SVM 
            SVM_k = fitcsvm(x_train_fold', l_train_fold', 'BoxConstraint', C_val, ...
                'KernelFunction', 'gaussian', 'KernelScale', sigma_val);
            
            % Validate 
            d_validate = SVM_k.predict(x_validate')';
            ind_correct = find(l_validate .* d_validate == 1);
            N_correct(k) = length(ind_correct);
        end
        
        P_correct(C_counter, sigma_counter) = sum(N_correct) / N_train;
    end
end

% Find optimal hyperparameters 
[max_correct, ind_i] = max(P_correct(:));
[ind_best_C, ind_best_sigma] = ind2sub(size(P_correct), ind_i);
C_best = C_list(ind_best_C);
sigma_best = sigma_list(ind_best_sigma);

% Display top 10 hyperparameter combinations 
[sorted_accuracies, sorted_indices] = sort(P_correct(:), 'descend');
fprintf('\nTop 10 Hyperparameter Combinations:\n');
fprintf('  Rank |   C      |  sigma   | Accuracy\n');
fprintf('-------|----------|----------|----------\n');
for rank = 1:min(10, length(sorted_accuracies))
    [C_idx, sigma_idx] = ind2sub(size(P_correct), sorted_indices(rank));
    fprintf('   %2d  | %.2e | %.2e | %.4f\n', rank, ...
        C_list(C_idx), sigma_list(sigma_idx), sorted_accuracies(rank));
end
fprintf('\n');

fprintf('Selected Optimal Hyperparameters:\n');
fprintf('C*: %.2e\n', C_best);
fprintf('Sigma*: %.2e\n', sigma_best);
fprintf('Best CV accuracy: %.4f (%.2f%%)\n\n', max_correct, 100*max_correct);

% Visualize cross-validation results 
figure(2), clf,
subplot(2,2,1),

% Accuracy vs C (averaged over sigma values)
P_correct_vs_C = mean(P_correct, 2);

plot(log10(C_list), P_correct_vs_C, 'k.-', 'LineWidth', 2, 'MarkerSize', 18);
hold on;

xlabel('log_{10} C');
ylabel('Average K-fold Validation Accuracy');
title('SVM: Accuracy vs Box Constraint C');
grid on;
plot(log10(C_best), P_correct_vs_C(ind_best_C), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('CV Accuracy', sprintf('C* = %.3g', C_best), 'Location', 'best');

subplot(2,2,2),
% Accuracy vs sigma (averaged over C values)
P_correct_vs_sigma = mean(P_correct, 1);
plot(log10(sigma_list), P_correct_vs_sigma, 'k.-', 'LineWidth', 2, 'MarkerSize', 18);
hold on;
xlabel('log_{10} sigma');
ylabel('Average K-fold Validation Accuracy');
title('SVM: Accuracy vs Kernel Scale Sigma');
grid on;
hold on;
plot(log10(sigma_best), P_correct_vs_sigma(ind_best_sigma), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('CV Accuracy', sprintf('\\sigma* = %.3g', sigma_best), 'Location', 'best');

% Train final SVM with optimal hyperparameters 
fprintf('Train final SVM with optimal hyperparameters: \n');
SVM_best = fitcsvm(x_train', labels_train', 'BoxConstraint', C_best, ...
    'KernelFunction', 'gaussian', 'KernelScale', sigma_best);

% Evaluate on test set
d_test_SVM = SVM_best.predict(x_test')';
ind_correct_SVM = find(labels_test .* d_test_SVM == 1);
ind_incorrect_SVM = find(labels_test .* d_test_SVM == -1);
p_error_SVM = length(ind_incorrect_SVM) / N_test;

fprintf('SVM Test Performance:\n');
fprintf('P(error) = %.4f \n', p_error_SVM);

% Visualize SVM results
subplot(2,2,[3,4]),
plot(x_test(1, ind_correct_SVM), x_test(2, ind_correct_SVM), 'g.', 'MarkerSize', 4); hold on,
plot(x_test(1, ind_incorrect_SVM), x_test(2, ind_incorrect_SVM), 'r.', 'MarkerSize', 4);

% Plot decision boundary
N_x = 200; N_y = 200;
x_grid = linspace(min(x_test(1,:)), max(x_test(1,:)), N_x);
y_grid = linspace(min(x_test(2,:)), max(x_test(2,:)), N_y);
[h, v] = meshgrid(x_grid, y_grid);
d_grid = SVM_best.predict([h(:), v(:)]);
z_grid = reshape(d_grid, N_y, N_x);
contour(x_grid, y_grid, z_grid, [0 0], 'k-', 'LineWidth', 2);
xlabel('x_1'), ylabel('x_2');
title(sprintf('SVM Test Results (P(error)=%.4f)', p_error_SVM));
legend('Correct', 'Incorrect', 'Decision Boundary', 'Location', 'best');
axis equal tight;

% PART 2: MULTI-LAYER PERCEPTRON (QUADRATIC ACTIVATION)
fprintf('PART 2: MLP WITH QUADRATIC ACTIVATION\n');
fprintf('Using K-fold cross-validation for model order selection\n\n');

% Convert labels to {1, 2} for MLP
labels_train_MLP = (labels_train + 3) / 2;  
labels_test_MLP = (labels_test + 3) / 2;

% MLP hyperparameter grid
P_values = [2, 4, 6, 8];  % Number of hidden perceptrons
n_classes = 2;                  % Binary classification

fprintf('Testing P values: ');
fprintf('%d ', P_values);
fprintf('\n\n');

% K-fold cross-validation for MLP 
fprintf('%d-fold cross-validation for MLP: \n', K);
cv_errors_MLP = zeros(1, length(P_values));

for j = 1:length(P_values)
    P = P_values(j);
    fprintf('P = %d: ', P);
    
    fold_errors = zeros(K, 1);
    
    for k = 1:K
        % Partition data 
        ind_validate = ind_partition_limits(k,1):ind_partition_limits(k,2);
        x_validate = x_train(:, ind_validate);
        l_validate = labels_train_MLP(ind_validate);
        
        if k == 1
            ind_train_fold = ind_partition_limits(k,2)+1:N_train;
        elseif k == K
            ind_train_fold = 1:ind_partition_limits(k,1)-1;
        else
            ind_train_fold = [1:ind_partition_limits(k,1)-1, ind_partition_limits(k,2)+1:N_train];
        end
        
        x_fold_train = x_train(:, ind_train_fold);
        l_fold_train = labels_train_MLP(ind_train_fold);
        
        % Train MLP with P perceptrons
        Mdl = trainCustomMLP(x_fold_train, l_fold_train, P, n_classes);
        
        % Validate
        predictions = predictCustomMLP(Mdl, x_validate);
        fold_errors(k) = sum(predictions ~= l_validate) / length(l_validate);
    end
    
    cv_errors_MLP(j) = mean(fold_errors);
    fprintf('CV error = %.4f\n', cv_errors_MLP(j));
end

% Select optimal P 
[min_error_MLP, best_idx_MLP] = min(cv_errors_MLP);
P_optimal = P_values(best_idx_MLP);

fprintf('Optimal P*: %d (CV error = %.4f)\n\n', P_optimal, min_error_MLP);

% Train final MLP with optimal P and multiple initializations
fprintf('Training final MLP with P* = %d: \n', P_optimal);
best_log_likelihood = -inf;
best_model_MLP = [];

for init = 1:3
    rng(5644 + init, 'twister');
    Mdl_final = trainCustomMLP(x_train, labels_train_MLP, P_optimal, n_classes);
    
    % Calculate training log-likelihood to select best initialization
    posterior_probs = mlpModelClassification(x_train, Mdl_final.params);
    log_likelihood = 0;
    for n_sample = 1:N_train
        true_class = labels_train_MLP(n_sample);
        log_likelihood = log_likelihood + log(posterior_probs(true_class, n_sample) + 1e-10);
    end
    
    if log_likelihood > best_log_likelihood
        best_log_likelihood = log_likelihood;
        best_model_MLP = Mdl_final;
    end
end

fprintf('Best model selected (log-likelihood = %.4f)\n\n', best_log_likelihood);

% Evaluate MLP on test set
predictions_test_MLP = predictCustomMLP(best_model_MLP, x_test);
p_error_MLP = sum(predictions_test_MLP ~= labels_test_MLP) / N_test;

fprintf('MLP Test Performance:\n');
fprintf('P(error) = %.4f (%.2f%%)\n', p_error_MLP, 100*p_error_MLP);

% Visualize MLP results
figure(3), clf,
subplot(1,2,1),
plot(P_values, cv_errors_MLP, 'b-o', 'LineWidth', 2, 'MarkerSize', 6); hold on,
plot(P_optimal, min_error_MLP, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Number of Perceptrons (P)');
ylabel('10-Fold CV Error');
title('MLP: Model Order Selection');
legend('CV Error', sprintf('P* = %d', P_optimal), 'Location', 'best');
grid on;

subplot(1,2,2),
ind_correct_MLP = find(predictions_test_MLP == labels_test_MLP);
ind_incorrect_MLP = find(predictions_test_MLP ~= labels_test_MLP);
plot(x_test(1, ind_correct_MLP), x_test(2, ind_correct_MLP), 'g.', 'MarkerSize', 4); hold on,
plot(x_test(1, ind_incorrect_MLP), x_test(2, ind_incorrect_MLP), 'r.', 'MarkerSize', 4);
% Plot MLP decision boundary
posterior_grid = mlpModelClassification([h(:)'; v(:)'], best_model_MLP.params);
[~, decision_grid] = max(posterior_grid, [], 1);
decision_grid = reshape(decision_grid, N_y, N_x);
contour(x_grid, y_grid, decision_grid, [1.5 1.5], 'k-', 'LineWidth', 2);
xlabel('x_1'), ylabel('x_2');
title(sprintf('MLP Test Results (P*=%d, P(error)=%.4f)', P_optimal, p_error_MLP));
legend('Correct', 'Incorrect', 'Decision Boundary', 'Location', 'best');
axis equal tight;

% FINAL COMPARISON
fprintf('FINAL COMPARISON: SVM vs MLP\n');
fprintf('  SVM (Gaussian kernel):\n');
fprintf('  C* = %.2e, sigma* = %.2e\n', C_best, sigma_best);
fprintf('  P(error) = %.4f (%.2f%%)\n', p_error_SVM, 100*p_error_SVM);
fprintf('\n  MLP (Quadratic activation):\n');
fprintf('  P* = %d perceptrons\n', P_optimal);
fprintf('  P(error) = %.4f (%.2f%%)\n\n', p_error_MLP, 100*p_error_MLP);

if p_error_SVM < p_error_MLP
    fprintf('Best Classifier: SVM (%.2f%% lower error)\n', 100*(p_error_MLP - p_error_SVM));
else
    fprintf('Best Classifier: MLP (%.2f%% lower error)\n', 100*(p_error_SVM - p_error_MLP));
end

% HELPER FUNCTIONS
% Data generation function 
function [x, labels] = generate_concentric_circles(N, r_minus1, r_plus1, sigma)
    % Equal priors for binary classes
    labels = 2 * (rand(1, N) >= 0.5) - 1;  % -1 or +1
    x = zeros(2, N);
    for i = 1:N
        if labels(i) == -1
            r = r_minus1;
        else
            r = r_plus1;
        end
        
        theta = -pi + 2*pi*rand();  % Uniform[-pi, pi]
        noise = sigma * randn(2, 1);  % Gaussian noise
        x(:, i) = r * [cos(theta); sin(theta)] + noise;
    end
end

% Custom MLP training function
function model = trainCustomMLP(X, Y, n_perceptrons, n_classes)
    [n_X, N] = size(X);
    n_Y = n_classes;
    
    % Xavier initialization
    params.A = randn(n_perceptrons, n_X) * sqrt(2/n_X);
    params.b = zeros(n_perceptrons, 1);
    params.C = randn(n_Y, n_perceptrons) * sqrt(2/n_perceptrons);
    params.d = zeros(n_Y, 1);
    
    % Convert to vector 
    vec_params_init = [params.A(:); params.b; params.C(:); params.d];
    size_params = [n_X; n_perceptrons; n_Y];
    
    % Optimize using fminsearch 
    n_params = length(vec_params_init);
    max_evals = max(5000, 50 * n_params);
    max_iter = max(2000, 20 * n_params);
    
    options = optimset('MaxFunEvals', max_evals, 'MaxIter', max_iter, ...
                       'TolFun', 1e-6, 'TolX', 1e-6, 'Display', 'off');
    
    vec_params = fminsearch(@(vec_params)(objectiveFunctionClassification(X, Y, size_params, vec_params)), ...
                           vec_params_init, options);
    
    % Convert back to structure 
    params.A = reshape(vec_params(1:n_X*n_perceptrons), n_perceptrons, n_X);
    params.b = vec_params(n_X*n_perceptrons+1:(n_X+1)*n_perceptrons);
    params.C = reshape(vec_params((n_X+1)*n_perceptrons+1:(n_X+1+n_Y)*n_perceptrons), n_Y, n_perceptrons);
    params.d = vec_params((n_X+1+n_Y)*n_perceptrons+1:(n_X+1+n_Y)*n_perceptrons+n_Y);
    
    model.params = params;
    model.n_perceptrons = n_perceptrons;
    model.n_classes = n_classes;
end

% MLP forward pass with quadratic activation and softmax
% Softmax implementation
% Numerical stability improvement 
function H = mlpModelClassification(X, params)
    N = size(X, 2);
    n_Y = length(params.d);
    
    % First layer
    U = params.A * X + repmat(params.b, 1, N);
    
    % Hidden layer: Quadratic activation 
    Z = activationFunctionQuadratic(U);
    
    % Second layer
    V = params.C * Z + repmat(params.d, 1, N);
    
    % Output layer - numerically stable softmax
    % Max-subtraction prevents overflow for large V values
    V_max = max(V, [], 1);
    E = exp(V - repmat(V_max, n_Y, 1));
    H = E ./ repmat(sum(E, 1), n_Y, 1);
end

% Quadratic activation function
function out = activationFunctionQuadratic(in)
    out = in.^2; 
end

% Cross-entropy objective function 
function obj_fnc_value = objectiveFunctionClassification(X, Y, size_params, vec_params)
    N = size(X, 2);
    n_X = size_params(1);
    n_perceptrons = size_params(2);
    n_Y = size_params(3);
    
    % Convert vector to structure
    params.A = reshape(vec_params(1:n_X*n_perceptrons), n_perceptrons, n_X);
    params.b = vec_params(n_X*n_perceptrons+1:(n_X+1)*n_perceptrons);
    params.C = reshape(vec_params((n_X+1)*n_perceptrons+1:(n_X+1+n_Y)*n_perceptrons), n_Y, n_perceptrons);
    params.d = vec_params((n_X+1+n_Y)*n_perceptrons+1:(n_X+1+n_Y)*n_perceptrons+n_Y);
    
    % Forward pass
    H = mlpModelClassification(X, params);
    
    % Cross-entropy loss 
    % ML estimation minimizes average cross-entropy loss
    cross_entropy = 0;
    for i = 1:N
        true_class = Y(i);
        % Add epsilon to avoid log(0)
        cross_entropy = cross_entropy - log(H(true_class, i) + 1e-10);
    end
    obj_fnc_value = cross_entropy / N;
end

% Prediction function
function predictions = predictCustomMLP(model, X)
    % Get posterior probabilities
    H = mlpModelClassification(X, model.params);
    
    % MAP decision rule
    [~, predictions] = max(H, [], 1);

end
