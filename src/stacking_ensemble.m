classdef stacking_ensemble
%%  Ensemble Learning Toolbox
%
%	A simple class/toolbox for creating custom classification and
%	regression ensemble models.
%
%--------------------------------------------------------------------------
%
%   Properties:
%       learner:
%           A function callback (Statistics and Machine Learning Toolbox
%           compatible)
%       model:
%           A trained meta model
%       ensemble:
%           A trained ensemble
%
%   Methods:
%       stacking_ensemble:
%           Creates an empty ensemble object
%       fit:
%           Trains the meta learner with provided data
%       predict:
%           Predict output from stacked ensemble
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
        
        model = {} % Trained meta (stacking) model
        learner = {} % Metal learner (callback)
        
        ensemble = {}; % Trained ensemble
    end
    
    methods
        
        % Class constructor
        function obj = stacking_ensemble(ensemble, learner)
            
            % Trained ensemble
            obj.ensemble = ensemble;
            
            % Meta learner
            obj.learner = learner;
        end
        
        % Fit stacking ensemble
        function obj = fit(obj, X, Y)
            
            % Predict models
            [~, x] = obj.ensemble.predict(X);
            
            % Train meta learner
            obj.model = obj.learner(x, Y);
                
        end
        
        % Predict new data
        function [Y] = predict(obj, X)
            
            % Predict with ensemble
            [~, x] = obj.ensemble.predict(X);
            
            % Predict with meta learner
            Y = obj.model.predict(x);
        end
    end
end

