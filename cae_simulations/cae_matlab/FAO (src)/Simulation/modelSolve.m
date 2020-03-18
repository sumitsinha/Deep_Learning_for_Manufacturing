% Solve model
function [U, GAP, flagSimulation, inputData]=modelSolve(data, stationData, paraID)

% Inputs:
% data: input model
% stationData: station data
% paraID: parameter ID

% Outputs:
% inputData: updated inputData field =[1xnStation]   
% U: deviation fields [nDoF, nStation]
    % Cumulative deviation at station[i]=sum(U(1 to i))
% GAP: gap field [1 x nStation]
% flagSimulation: simulation flag [1 x nStation]
% inputData: input data fields [1 x nStation]
%----------------

% log file
filelog='log_sol_training.txt';
fclose('all');
%------------------------------

% write log file
idlog=fopen(filelog,'a');
fprintf(idlog,'Training sample [%g]\r\n',paraID);
fclose(idlog); 

                % STEP 0 - set UCS (if necessary)
                nStation=length(stationData);
                for stationID=1:nStation
                    data=modelAssignUCSLocatorPlacement(data, stationData, stationID);
                end
% STEP 1 - set parameter ID
data.Assembly.X.ID=paraID; data.Assembly.X.nD=1;
% STEP 2 - assign parameters to model
data=modelAssignParameters(data);
% STEP 3 - build all inputs
data=modelBuildInput(data,[1 0 0]);
% STEP 4 - solve model
[U, GAP, flagSimulation, inputData]=modelStationSolve(data, stationData);
    
