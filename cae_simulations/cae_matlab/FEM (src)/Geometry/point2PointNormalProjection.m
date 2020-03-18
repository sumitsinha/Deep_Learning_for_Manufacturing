function [Pmp, Nmp, flag]=point2PointNormalProjection(fem, Pm, idpart, SearchDist)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model
% * Pm: point (xyz) to be projected on the surface - [n,3]
% * idpart: part ID (integer)
% * SearchDist: searching distance to compute projection (double)
%
% Outputs:
% * Pmp: projected point (xyz) - [n,3]
% * Nmp: normal vector corresponding to the projected point (xyz) - [n,3]
% * flag:
    % true: point projected
    % false: failed to project point
%
% compile: mex point2PointNormalProjection.cpp preProcessingLib.cpp shapeFcn.cpp  -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

