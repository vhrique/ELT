classdef custom_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom classification ensemble
%	models.
%
%--------------------------------------------------------------------------
%
%   Version 0.1 - Copyright 2018
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
        models
        learners
        features = {}
        
        combination_method = 'majority'
        stacking_model
        stacking_learner
    end
    
    methods
        
        % Create object
        function obj = custom_ensemble(learners, comb, stacker, features)
            
            % Sanity check
            if nargin == 0
                error('No parameters')
            end
            
            % Initialize learners list
            if nargin > 0
                obj.learners = learners;
            end
            
            % Define combination method
            if nargin > 1
                obj.combination_method = comb;
            end
            
            % Define stacker learner
            if nargin > 2
                obj.stacking_learner = stacker;
            end
            
            % Initialize features list
            if nargin > 3
                if length(features) ~= length(obj.learners)
                    error('Wrong features list size')
                end                
                obj.features = features;
            end
        end
        
        % Fit ensemble
        function obj = fit(obj, X, Y)
            
            % Sanity check
            if nargin < 3
                error('No data')
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
            
            % Fit stacking model
            if strcmp(obj.combination_method, 'stacking')
                
                % Predict for all models
                for i = 1 : length(obj.models)
                    y(:,i) = predict(obj.models{i}, X(:,obj.features{i}));
                end
                
                % Fit stacking model
                obj.stacking_model = obj.stacking_learner((2*y-1), Y);
            end
        end
        
        % Predict new data
        function Y = predict(obj, X)
            
            % Sanity check
            if nargin < 2
                error('No data')
            end
            
            % Predict for all models
            for i = 1 : length(obj.models)
                y(:,i) = predict(obj.models{i}, X(:,obj.features{i})) >= 0.5;
            end
            
            % Combine predictions
            switch obj.combination_method
                case 'majority'
                    Y = sum(y, 2) >= size(y, 2)/2;
                    
                case 'stacking'
                    Y = predict(obj.stacking_model, (2*y-1));
            end
        end
    end
    
end

