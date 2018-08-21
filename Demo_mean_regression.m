%%  Ensemble Learning Toolbox - Regression Demo
%
%	A simple average ensemble regression example.
%
%--------------------------------------------------------------------------
%
%   Version 0.X - Copyright 2018
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

%% Ensemble

% Initialize Ensemble
ens = custom_ensemble;
ens.learners = {linear_svm, gaussian_svm, tree};
ens.meta_learner = {}; % this implies that mean is used

% Train Ensemble
ens = ens.fit(X_train, Y_train);

% Predict
y_ens = ens.predict(X_test);

% Rescale target
Y_test = Y_test * std_y + mean_y;
y_ens = y_ens * std_y + mean_y;

% Compute R squared
R2 = 1 - sum((Y_test - y_ens) .^ 2) / sum((Y_test - mean(Y_test)) .^ 2);

% Print R Squared
fprintf("---------------------\n");
fprintf("Ensemble R²: %.3f\n", R2);
fprintf("---------------------\n");

% Plot result
plot(Y_test, 'b'); hold on;
plot(y_ens, 'r'); hold off;