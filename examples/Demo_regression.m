%%  Ensemble Learning Toolbox - Regression Demo
%
%	A simple ensemble regression example.
%
%--------------------------------------------------------------------------
%
%   Version 1.X - Copyright 2020
%
%       For new releases and bug fixing of this Tool Set please send e-mail
%       to the authors.
%
%--------------------------------------------------------------------------
%
%   Institution:
%       Optimization, Modeling and Control Systems Research Group
%
%       Graduate Program in Industrial and Systems Engineering - PPGEPS
%
%       Pontifical Catholic University of Paraná - Brazil.
%           <http://en.pucpr.br/>
%
%--------------------------------------------------------------------------
%
%	Authors:
%       Victor Henrique Alves Ribeiro
%           <victor.henrique@pucpr.edu.br>
%

%% Data preparation

% Load data
load examgrades.mat
X = grades(:, 1:4);
Y = grades(:, 5);

% Split train and test data
[train, test] = dividerand(length(Y), 0.7, 0.3);
X_train = X(train, :);
Y_train = Y(train);
X_test = X(test, :);
Y_test = Y(test);

% Scale data
mean_x = mean(X_train);
std_x = std(X_train);

mean_y = mean(Y_train);
std_y = std(Y_train);

X_train = (X_train - mean_x) ./ std_x;
X_test = (X_test - mean_x) ./ std_x;

Y_train = (Y_train - mean_y) ./ std_y;
Y_test = (Y_test - mean_y) ./ std_y;

% Prepare learners
linear_svm = @(x, y)fitrsvm(x, y, 'KernelFunction', 'linear');
gaussian_svm = @(x, y)fitrsvm(x, y, 'KernelFunction', 'gaussian');
tree = @(x, y)fitrtree(x, y);
learners = {linear_svm, gaussian_svm, tree};

%% Single model

mdl = linear_svm(X_train, Y_train); % Train Ensemble
y_ens = mdl.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Single Model R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Basic Ensemble

ens = regression_ensemble(learners); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Ensemble R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Bagged Ensemble

ens = regression_ensemble({linear_svm}); % Initialize Ensemble
ens = ens.fit_bag(X_train, Y_train, 50, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Bagging R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Boosted Ensemble

ens = regression_ensemble({linear_svm}); % Initialize Ensemble
ens = ens.fit_boost(X_train, Y_train, 50); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Boosting R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Random Subspace Ensemble

ens = regression_ensemble({linear_svm}); % Initialize Ensemble
ens = ens.fit_sub(X_train, Y_train, 50, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Random Subspace R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Random Forest Ensemble

ens = regression_ensemble({linear_svm}); % Initialize Ensemble
ens = ens.fit_rf(X_train, Y_train, 50, 0.5, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Random Forest R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");

%% Stacked Ensemble

ens = stacking_ensemble(ens, linear_svm); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Rescale target
y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Print R Squared
fprintf("---------------------\n");
fprintf("Stacking R²: %.3f\n", ...
    1 - sum((y_test - y_ens) .^ 2) / sum((y_test - mean(y_test)) .^ 2));
fprintf("---------------------\n");