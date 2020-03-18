% Transform all data into global coordinate frame
function fem=femGlobalSolution(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model
%
% Outputs:
% * fem: updated fem model with:
    % .Sol.U - solution vector
    % .Sol.R - reaction forces vector
%
% compile: mex femGlobalSolution.cpp -largeArrayDims
% Note: use "mex -g" to run in debug mode