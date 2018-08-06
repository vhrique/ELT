classdef custom_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom binary classification and
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
        
        mode = {}; % Ensemble's mode {regression, classification}
    end
    
    methods
        
        % Fit ensemble
        function obj = fit(obj, X, Y)
            
            % Sanity check
            if nargin < 3
                error('No data')
            end
            
            % Check if classification or regression problem
            if islogical(Y)
                obj.mode = 'classification';
            else
                obj.mode = 'regression';
            end
            
            % If no features have been defined, use all
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
                
                % Predict for all models
                y = zeros(size(X, 1), length(obj.models)); % Initialize
                for i = 1 : length(obj.models)
                    y(:,i) = predict(obj.models{i}, X(:, obj.features{i}));
                end
                y = y * 1.0; % Convert to float
                    
                % Fit stacking model
                obj.meta_model = obj.meta_learner(y, Y);
            end
        end
        
        % Predict new data
        function Y = predict(obj, X)
            
            % Sanity check
            if nargin < 2
                error('No data')
            end
            
            % Predict for all models
            y = zeros(size(X, 1), length(obj.models)); % Initialize matrix
            for i = 1 : length(obj.models)
                y(:,i) = predict(obj.models{i}, X(:, obj.features{i}));
            end
            y = y * 1.0; % Convert to float                
            
            % Combine predictions
            if isempty(obj.meta_model)
                Y = mean(y, 2);
                if strcmp(obj.mode, 'classification'), Y = Y >= 0.5; end
            else
                Y = predict(obj.meta_model, y);
            end
        end
    end
    
end

