function [quad, tria, node]=readMeshNas(filename)
%
% ARGUMENTs:
%
% Inputs:
% * filename: mesh file to be imported
%
% Outputs:
% * quad - quad element(nx3). If no element => -1. IDS are re-numbered starting from ID=1
% * tria - tria element(nx3). If no element => -1. IDS are re-numbered starting from ID=1
% * node - xyz coordinates of nodes If no node => -1
%
% compile: readMeshNas.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

    
