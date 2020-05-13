classdef regression_ensemble < base_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom classification and
%	regression ensemble models.
%
%--------------------------------------------------------------------------
%
%   Properties inherited from base_ensemble
%
%   Methods:
%       regression_ensemble:
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
    
    methods
        
        % Constructor
        function obj = regression_ensemble(learners)
            obj = obj@base_ensemble(learners);
        end
        
        % Train through boosting
        function obj = fit_boost(obj, X, Y, N)
            
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
                    loss = abs(Y - y_pred);
                    loss = loss / max(loss);
                    average_loss = sum(loss .* D);
                    
                    % Predictor confidence
                    confidence = average_loss/(1 - average_loss);
                    
                    % Adjust sample weights
                    D = D .* (confidence .^ (1 - loss));
                    D = D / sum(D);
                    
                    % Model weight
                    obj.weights(N*(i-1)+n,1) = log(1/confidence);
                
                    % Store features (all)
                    obj.features{N*(i-1)+n} = true(size(X, 2), 1);
                end
            end
        end
        
        % Predict new data
        function [Y, y] = predict(obj, X)

            % Initialize output array
            y = zeros(size(X,1), length(obj.models));
                
            % For each model
            for i = 1 : length(obj.models)

                % Predict
                y(:,i) = predict(obj.models{i}, X(:, obj.features{i}));
            end             
            
            % Combine predictions
            Y = y * obj.weights / sum(obj.weights);
        end
    end
end

