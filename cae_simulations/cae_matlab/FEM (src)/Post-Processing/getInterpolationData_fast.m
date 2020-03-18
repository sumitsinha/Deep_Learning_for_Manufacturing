% Interpolate FEM model
function [Pmp, data, flag]=getInterpolationData_fast(fem)
%
% ARGUMENTs:
%
% Inputs:
% * fem model
    % fem.Post.Interp.Pm: points to be projected (nx3)
    % fem.Post.Interp.Domain: domain (integer)
    % fem.Post.Interp.SearchDist: (double)
    % fem.Post.Interp.InterpVariable: interpolation variable ("u", "v",..."gap", "user")
    % fem.Post.Interp.ContactPair: ID contact pair (integer) to process "gap" variable. If no pair, then set "0"
%
% Outputs:
% * Pmp - projected points (nx3)
% * data - interpolated data (1xn)
% * flag - true/false (1xn)
%
% compile: mex getInterpolationData_fast.cpp preProcessingLib.cpp shapeFcn.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
    % Note: use "mex -g" to run in debug mode
    
