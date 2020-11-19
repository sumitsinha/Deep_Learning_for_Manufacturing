% Solve "Clamp" at station level
function [data, U, GAP, Fa, flagSimulation]=modelStationClamp(data, stationID)

% Inputs:
% data: model
% stationData(stationID): station data
% stationID: station ID (integer)
        
% Outputs:
% data: udpated model
% U: deviation field [x, y, z] (nnode, 3)  
% GAP: gap field [1 x no. contact pairs]
% Fa: reaction forces
% flagSimulation: simulation flag

% Init output
U=[]; %#ok<NASGU>
GAP=[]; %#ok<NASGU>
Fa=[]; %#ok<NASGU>
%
% Get part definitions
[pdata, flag]=getPartDescription(data);
if ~flag
    error('Failed to calculate part descriptors @Station[%g]', stationID)
end
%
% Reset solution
data=modelReset(data, false, true); % reset "assembly" solution
%
% Solve model
sdata=[];
[data, ~, flagsim, Fa]=simulationCore(data, sdata, pdata(:,1)', 'refresh');
flagm=data.Assembly.Log{1}.Done; % solution reached
%
% Save back deviation field
U=data.Assembly.U{1};
GAP=data.Assembly.GAP{1};
%
if flagsim && flagm
    flagSimulation=true;
else
    flagSimulation=false;
end
%