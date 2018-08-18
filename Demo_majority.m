%%  Ensemble Learning Toolbox - Majority Voting Demo
%
%	A simple majority voting ensemble example.
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
load fisheriris.mat
X = meas;
Y = categorical(species);

% Split train and test data
[train, test] = dividerand(length(Y), 0.7, 0.3);
X_train = X(train, :);
Y_train = Y(train);
X_test = X(test, :);
Y_test = Y(test);

% Prepare learners
linear = templateSVM('KernelFunction','linear');
linear_svm = @(x, y)fitcecoc(x, y, 'Learners', linear);
gaussian = templateSVM('KernelFunction','gaussian');
gaussian_svm = @(x, y)fitcecoc(x, y, 'Learners', gaussian);
knn1 = @(x, y)fitcknn(x, y, 'NumNeighbors', 1);
knn3 = @(x, y)fitcknn(x, y, 'NumNeighbors', 3);
knn5 = @(x, y)fitcknn(x, y, 'NumNeighbors', 5);
tree = @(x, y)fitctree(x, y);

%% Ensemble

% Initialize Ensemble
ens = custom_ensemble;
ens.learners = {linear_svm, gaussian_svm, knn1, knn3, knn5, tree};
ens.meta_learner = {}; % this implies that majority voting is used

% Train Ensemble
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