function [Pmp, flag, gsign, edata]=pointNormal2PointProjection(fem, Pm, Nm, idpart)
%
% ARGUMENTs:
%
% Inputs:
% * fem: input fem model. The numerical error which control if the projected point is inside given mesh element is controlled by:
    % .Options.Eps: (double)
% * Pm: point (xyz) to be projected on the surface - [n,3]
% * Nm: direction (xyz) of projection - [n,3]
% * idpart: part ID (integer)
%
% Outputs:
% * Pmp: projected point (xyz) - [n,3]
% * flag:
    % true: point projected
    % false: failed to project point
% * gsign: signed distance between the Pm and the surface
% * edata=[n,2]
    % (1) element ID
    % (2) element type
        % 1=> QUAD
        % 2=> TRIA
%
% compile: mex pointNormal2PointProjection.cpp preProcessingLib.cpp shapeFcn.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

