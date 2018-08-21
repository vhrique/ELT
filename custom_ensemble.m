classdef custom_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom classification and
%	regression ensemble models.
%
%--------------------------------------------------------------------------
%
%   Properties:
%       models:
%           A list of trained models
%       learners:
%           A list of function callbacks (Statistics and Machine Learning
%           Toolbox)
%       features:
%           A list of features to be used by each model
%       stacking_learner:
%           A function callback (Statistics and Machine Learning Toolbox)
%
%   Methods:
%       custom_ensemble:
%           Creates an empty ensemble object
%       fit:
%           Trains the learners with provided data
%       predict:
%           Predicts the output for new data with the ensemble
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
    
    properties

        models = {} % Trained models' list
        learners = {} % Learners' list (callbacks)
        features = {} % Features' list
        
        meta_model = {} % Trained meta (stacking) model
        meta_learner = {} % Metal learner (callback)
        
        mode = {}; % Ensemble's mode {classification, regression}
        classes = {}; % Problem's classes
    end
    
    methods
        
        % Fit ensemble
        function obj = fit(obj, X, Y)
            
            % Check if classification or regression problem
            if iscategorical(Y)
                obj.mode = 'classification';
                obj.classes = unique(Y)';
            else
                obj.mode = 'regression';
                obj.classes = {'num'};
            end
            
            % If no features have been defined, use all of them
            if isempty(obj.features)
                for i = 1 : length(obj.learners)
                    obj.features{i} = 1:size(X, 2);
                end
            end
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                obj.models{i} = obj.learners{i}(X(:,obj.features{i}), Y);
            end
            
            % Fit stacking meta model
            if ~isempty(obj.meta_learner)
                
                % Check number of classes, models and outputs
                n_classes = length(obj.classes);
                n_models = length(obj.models);
                n_outputs = length(Y);
                
                % Initialize classes matrix
                class_matrix = repmat(obj.classes, n_outputs, 1);
                
                % Initialize output array
                y = zeros(n_outputs, n_classes * n_models);
                
                % For each model
                for i = 1 : n_models
                    
                    % Predict
                    y_aux = predict(obj.models{i}, X(:, obj.features{i}));
                    
                    % If classification task, adjust outputs
                    if strcmp(obj.mode, 'classification')                        
                        y(:, n_classes*(i-1)+1 : n_classes*i) = 1.0 * ...
                            (repmat(y_aux, 1, n_classes) == class_matrix);
                    else
                        y(:,i) = y_aux;
                    end
                end
                    
                % Fit stacking model
                obj.meta_model = obj.meta_learner(y, Y);
            end
        end
        
        % Predict new data
        function Y = predict(obj, X)
                
            % Check number of classes, models and outputs
            n_classes = length(obj.classes);
            n_models = length(obj.models);
            n_outputs = size(X, 1);
            
            % Initialize classes matrix
            class_matrix = repmat(obj.classes, n_outputs, 1)';

            % Initialize output array
            y = zeros(n_outputs, n_classes * n_models);
                
            % For each model
            for i = 1 : n_models

                % Predict
                y_aux = predict(obj.models{i}, X(:, obj.features{i}));

                % If classification task, adjust outputs
                if strcmp(obj.mode, 'classification')                        
                    y(:, n_classes*(i-1)+1 : n_classes*i) = 1.0 * ...
                        (repmat(y_aux, 1, n_classes) == class_matrix');
                else
                    y(:,i) = y_aux;
                end
            end             
            
            % Combine predictions
            if isempty(obj.meta_model)
                
                if strcmp(obj.mode, 'classification')
                    % Initialize output
                    Y = zeros(n_outputs, n_classes);

                    % Sum prediction occurrences
                    for i = 1 : n_models
                        Y = Y + y(:, n_classes*(i-1)+1:n_classes*i);
                    end

                    % Vote
                    [~, idx] = max(Y, [], 2);
                    Y = class_matrix(idx);
                else
                    Y = mean(y, 2);
                end
            else
                % Stacking
                Y = predict(obj.meta_model, y);
            end
        end
    end
end

