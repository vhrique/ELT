%%  Ensemble Learning Toolbox - Classification Demo
%
%	A simple classification ensemble example.
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

% Load and adjust data
load fisheriris.mat
X = meas; % Features
Y = categorical(species); % Outputs - converted to categorical

% Split train and test data
[train, test] = dividerand(length(Y), 0.7, 0.3);
X_train = X(train, :);
Y_train = Y(train);
X_test = X(test, :);
Y_test = Y(test);

% Adjust data
X_test = (X_test - mean(X_train)) ./ std(X_train);
X_train = (X_train - mean(X_train)) ./ std(X_train);

% Prepare learners
linear = templateSVM('KernelFunction','linear');
linear_svm = @(x, y)fitcecoc(x, y, 'Learners', linear);
gaussian = templateSVM('KernelFunction','gaussian');
gaussian_svm = @(x, y)fitcecoc(x, y, 'Learners', gaussian);
knn1 = @(x, y)fitcknn(x, y, 'NumNeighbors', 1);
knn3 = @(x, y)fitcknn(x, y, 'NumNeighbors', 3);
knn5 = @(x, y)fitcknn(x, y, 'NumNeighbors', 5);
tree = @(x, y)fitctree(x, y);
learners = {linear_svm, gaussian_svm, knn1, knn3, knn5, tree};

%% Single Model

mdl = tree(X_train, Y_train); % Train Model
y_ens = mdl.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Decision Tree: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% Basic Ensemble with all learners

ens = classification_ensemble(learners); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Ensemble: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% Bagged Ensemble

ens = classification_ensemble({tree}); % Initialize Ensemble
ens = ens.fit_bag(X_train, Y_train, 50, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Bagging: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% Boosted Ensemble

ens = classification_ensemble({tree}); % Initialize Ensemble
ens = ens.fit_boost(X_train, Y_train, 50); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Boost: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% Random Subspace Ensemble

ens = classification_ensemble({tree}); % Initialize Ensemble
ens = ens.fit_sub(X_train, Y_train, 50, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Randon Subspace: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% "Random Forest" Ensemble

ens = classification_ensemble({tree}); % Initialize Ensemble
ens = ens.fit_rf(X_train, Y_train, 50, 0.5, 0.5); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Random Forest: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");

%% Stacked Ensemble

ens = stacking_ensemble(ens, tree); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict

% Print result
fprintf("---------------------\n");
fprintf("Stacking: %.2f%%\n", ...
    100 * sum(y_ens == Y_test) / length(Y_test));
fprintf("---------------------\n");