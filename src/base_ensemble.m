classdef base_ensemble
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
%           Toolbox compatible)
%       features:
%           A list of features to be used by each model
%       weights:
%           A list of weights to be used by each model
%
%   Methods:
%       base_ensemble:
%           Creates an empty ensemble object
%       fit:
%           Trains the learners with provided data
%       fit_bag:
%           Trains the learners through bootstrap aggregating
%       fit_sub:
%           Trains the learners through random subspace
%       fit_rf:
%           Trains the learners through bootstrap aggregating and random
%           subspace
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

        learners = {} % Learners' list (callbacks)
        
        models = {} % Trained models' list
        features = {} % Features' list
        weights = [] % Model's weights list
    end
    
    methods
        
        % Class constructor
        function obj = base_ensemble(learners)
            obj.learners = learners;
        end
        
        % Fit ensemble
        function obj = fit(obj, X, Y)
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                
                % Store features (all)
                obj.features{i} = true(size(X, 2), 1);
                
                % Train model
                obj.models{i} = obj.learners{i}(X, Y);
            end
            
            % Store weights
            obj.weights = ones(length(obj.models),1);
        end
        
        % Fit bagging
        function obj = fit_bag(obj, X, Y, N, R)
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                for n = 1 : N
                    
                    % Boostrap samples with replacement
                    k = round(length(Y) * R);
                    train = datasample(1:length(Y), k);
                
                    % Store features (all)
                    obj.features{N*(i-1)+n} = true(size(X, 2), 1);
                    
                    % Train models
                    obj.models{N*(i-1)+n} = ...
                        obj.learners{i}(X(train,:), Y(train));
                end
            end
            
            % Store weights
            obj.weights = ones(length(obj.models),1);
        end
        
        % Fit random subspace
        function obj = fit_sub(obj, X, Y, N, F)
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                for n = 1 : N
                    
                    % Select random features
                    while true
                        
                        % Select features
                        obj.features{N*(i-1)+n} = rand(1,size(X,2)) <= F;
                        
                        % Stop if Ok
                        if sum(obj.features{N*(i-1)+n}) > 0
                            break;
                        end
                    end
                    
                    % Fit models
                    obj.models{N*(i-1)+n} = ...
                        obj.learners{i}(X(:, obj.features{N*(i-1)+n}), Y);
                end
            end
            
            % Store weights
            obj.weights = ones(length(obj.models),1);
        end
        
        % Fit "random forest" (bagging plus random subspace)
        function obj = fit_rf(obj, X, Y, N, R, F)
            
            % Fit all learners
            for i = 1 : length(obj.learners)
                for n = 1 : N
                    
                    % Boostrap samples with replacement
                    k = round(length(Y) * R);
                    train = datasample(1:length(Y), k);
                    
                    % Select random features
                    while true
                        
                        % Select features
                        obj.features{N*(i-1)+n} = rand(1,size(X,2)) <= F;
                        
                        % Stop if Ok
                        if sum(obj.features{N*(i-1)+n}) > 0
                            break;
                        end
                    end
                    
                    % Fit models
                    obj.models{N*(i-1)+n} = ...
                        obj.learners{i}(...
                        X(train, obj.features{N*(i-1)+n}), ...
                        Y(train));
                end
            end
            
            % Store weights
            obj.weights = ones(length(obj.models),1);
        end
    end
end

