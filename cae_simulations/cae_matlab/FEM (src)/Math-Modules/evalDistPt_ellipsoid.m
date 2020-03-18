% This function is part of the morphing mesh method as is used to evaluated normalised distance of "Point" within given ellipsoid and control point
function dist=evalDistPt_ellipsoid(Point, radius, Pc, C, RotC)
%
% ARGUMENTs:
%
% Inputs:
% Point: point to be evauated (m,3)
% radius: radius of the ellipsoid (1x3)
% Pc: control point of morphing mesh model
% C: centre of the ellipsoid  (xyz)
% RotC: rotation matrix of the ellipsoid (3x3)
%
% Outputs:
% dist: projected point (xyz) - (m,3)
%
% compile: mex evalDistPt_ellipsoid.cpp preProcessingLib.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

