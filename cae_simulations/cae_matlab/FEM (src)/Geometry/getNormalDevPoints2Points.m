% Find closest deviation between points "Pp" and point "Pn". Signed distance are computed along "Nm"
function dev=getNormalDevPoints2Points(Pm, Nm, Pp, dsearN, dsearT)
%
% ARGUMENTs:
%
% Inputs:
% Pm: reference points (nx3) 
% Nm: normal vectors of Pm (nx3)
% Pp: points to be projected (mx3)
% dsear => cylindrical serching distance
    % N: normal searching distance (double)
    % T: tangent searching distance (double)
%
% Outputs:
% dev: vector od signed deviations (mx1). For every point "Pp" the
% correspoding signed deviation is the average of all signed distance which satisfy the condition abs(gsign)<=dsearN & abs(gsign)<=dsearT
%
% compile: mex getNormalDevPoints2Points.cpp preProcessingLib.cpp -largeArrayDims COMPFLAGS="/openmp $COMPFLAGS"
% Note: use "mex -g" to run in debug mode

    
