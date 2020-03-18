% Renumber nodes ID and elements connection starting from ID=1
function elementr=renumberElements(element, idnode)
%
% ARGUMENTs:
%
% Inputs:
% element: mesh connectivity (m,3 or 4) => depending on QUAd or TRIA meshes
% idnode: nodes IDs (nx1)
%
% Outputs:
% elementr: updated mesh connectivity (m,3 or 4) => depending on QUAd or TRIA meshes
%
% compile: mex renumberElements.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

