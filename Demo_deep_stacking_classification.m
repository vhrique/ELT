%%  Ensemble Learning Toolbox - "Deep" Stacking Demo
%
%	A two-layer stacking ensemble example.
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

% Load and adjust data
load ionosphere.mat
Y = categorical(Y);

% Split train and test data
[train, test] = dividerand(length(Y), 0.7, 0.3);
X_train = X(train, :);
Y_train = Y(train);
X_test = X(test, :);
Y_test = Y(test);

% Prepare learners
linear_svm = @(x, y)fitcsvm(x, y, 'KernelFunction', 'linear');
gaussian_svm = @(x, y)fitcsvm(x, y, 'KernelFunction', 'gaussian');
knn1 = @(x, y)fitcknn(x, y, 'NumNeighbors', 1);
knn3 = @(x, y)fitcknn(x, y, 'NumNeighbors', 3);
knn5 = @(x, y)fitcknn(x, y, 'NumNeighbors', 5);
tree = @(x, y)fitctree(x, y);

%% Ensemble

% Initialize first layer ensembles
ens1 = custom_ensemble;
ens1.learners = {linear_svm, knn1, knn5};
ens1.meta_learner = {}; % this implies that majority voting is used
ensl_1 = @(x, y)fit(ens1, x, y); % create learner

ens2 = custom_ensemble;
ens2.learners = {gaussian_svm, knn3, tree};
ens2.meta_learner = linear_svm; % this implies that stacking is used
ensl_2 = @(x, y)fit(ens2, x, y); % create learner

ens3 = custom_ensemble;
ens3.learners = {linear_svm, knn1, tree};
ens3.meta_learner = {}; % this implies that majority voting is used
ensl_3 = @(x, y)fit(ens3, x, y); % create learner

% Initialize second layer ensemble
ens = custom_ensemble;
ens.learners = {ensl_1, ensl_2, ensl_3};
ens.meta_learner = gaussian_svm; % this implies that stacking is used

% Train ensemble
ens = ens.fit(X_train, Y_train);

% Predict
y_ens = ens.predict(X_test);

% Compute confusion matrix and compute accuracy
c = confusionmat(Y_test, y_ens);
acc = sum(sum(c.*eye(size(c))))/sum(sum(c));

% Print result
fprintf("---------------------\n");
fprintf("Ensemble: %f%%\n", acc * 100);
fprintf("---------------------\n");