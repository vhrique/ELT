classdef classification_ensemble < base_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom classification and
%	regression ensemble models.
%
%--------------------------------------------------------------------------
%
%   Properties:
%       classes:
%           A list of the problem's classes
%
%       *other properties inherited from base_ensemble
%
%   Methods:
%       classification_ensemble:
%           Creates an empty ensemble object
%       fit_boost:
%           Trains the learners through boosting
%       predict:
%           Predicts outputs
%
%       *other methods inherited from base_ensemble
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
    
    properties(Access = protected)
        classes = {}; % Problem's classes
    end
    
    methods
        
        % Constructor
        function obj = classification_ensemble(learners)
            obj = obj@base_ensemble(learners);
        end
        
        % Fit ensemble
        function obj = fit(obj, X, Y)
            
            % Get classes
            obj.classes = unique(Y)';
            
            % Train models
            obj = fit@base_ensemble(obj, X, Y);
        end
        
        % Fit bagging
        function obj = fit_bag(obj, X, Y, N, R)
            
            % Get classes
            obj.classes = unique(Y)';
            
            % Train models
            obj = fit_bag@base_ensemble(obj, X, Y, N, R);
        end
        
        % Fit subspace
        function obj = fit_sub(obj, X, Y, N, F)
            
            % Get classes
            obj.classes = unique(Y)';
            
            % Train models
            obj = fit_sub@base_ensemble(obj, X, Y, N, F);
        end
        
        % Fit bagging with subspace
        function obj = fit_rf(obj, X, Y, N, R, F)
            
            % Get classes
            obj.classes = unique(Y)';
            
            % Train models
            obj = fit_rf@base_ensemble(obj, X, Y, N, R, F);
        end
        
        % Train through boosting
        function obj = fit_boost(obj, X, Y, N)
            
            % Get classes
            obj.classes = unique(Y)';
            
            % Initialize sample weights
            D = repmat(1/length(Y), length(Y), 1);
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                for n = 1 : N
                    
                    % Select samples
                    use = randsrc(length(Y),1,[1:length(Y); D']);
                    
                    % Train model
                    obj.models{N*(i-1)+n} = obj.learners{i}(X(use,:), Y(use));
                    
                    % Predict
                    y_pred = predict(obj.models{N*(i-1)+n}, X);
                    
                    % Compute error
                    total_error = sum(y_pred ~= Y) / length(Y);
                    total_error = max(0.01, min(total_error, 0.99));
                    
                    % Compute amount of say
                    amount_of_say = log((1 - total_error)/total_error)/2;
                    
                    % Adjust sample weights
                    D(y_pred ~= Y) = D(y_pred ~= Y) * exp(amount_of_say);
                    D(y_pred == Y) = D(y_pred == Y) * exp(-amount_of_say);
                    D = D / sum(D);
                    
                    % Model weight
                    obj.weights(N*(i-1)+n) = amount_of_say;
                
                    % Store features (all)
                    obj.features{N*(i-1)+n} = true(size(X, 2), 1);
                end
            end
        end
        
        % Predict new data
        function [Y, y] = predict(obj, X)
                
            % Check number of classes, models and outputs
            n_classes = length(obj.classes);
            n_models = length(obj.models);
            n_samples = size(X, 1);
            
            % Initialize classes matrix
            class_matrix = repmat(obj.classes, n_samples, 1)';

            % Initialize output array
            y_all = zeros(n_samples, n_classes); % all models
            y = zeros(n_samples, n_classes * n_models); % by model
                
            % For each model
            for i = 1 : n_models

                % Predict
                y_aux = predict(obj.models{i}, X(:, obj.features{i}));

                % Adjust outputs
                y(:, n_classes*(i-1)+1 : n_classes*i) = ...
                    obj.weights(i) * ...
                    (repmat(y_aux, 1, n_classes) == class_matrix');
                y_all = y_all + ...
                    obj.weights(i) * ...
                    (repmat(y_aux, 1, n_classes) == class_matrix');
                
            end

            % Vote
            [~, idx] = max(y_all, [], 2);
            Y = class_matrix(idx);
                    
        end
    end
end

