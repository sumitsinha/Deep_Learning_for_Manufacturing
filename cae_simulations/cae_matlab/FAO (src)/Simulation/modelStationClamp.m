% Solve "Clamp" at station level
function [data, U, GAP, Ka, flagSimulation]=modelStationClamp(data, stationID)

% Inputs:
% data: model
% stationData(stationID): station data
% stationID: station ID (integer)
        
% Outputs:
% data: udpated model
% U: deviation field [x, y, z] (nnode, 3)  
% GAP: gap field [1 x no. contact pairs]
% Ka: assembly stiffness matrix (ndof x ndof) - sparse matrix
% flagSimulation: simulation flag

% Init output
U=[]; %#ok<NASGU>
GAP=[]; %#ok<NASGU>
Ka=[]; %#ok<NASGU>
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
[data, flagsim, Ka]=runSimulationCore(data, sdata, pdata, 'refresh');
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