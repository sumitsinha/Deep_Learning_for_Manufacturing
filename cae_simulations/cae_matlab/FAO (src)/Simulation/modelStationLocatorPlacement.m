% Solve "locator placement" at station level
function [data, U, GAP]=modelStationLocatorPlacement(data, stationData, stationID)

% Inputs:
% data: model
% stationData(stationID): station data
% stationID: station ID (integer)
        
% Outputs:
% data: udpated model model with geometry and inputs
        % refresh existing geometry
        % do not update stiffness matrix
        % do not recompute part UCS
        % only refresh part features, such as hole and slots
% U: deviation field [x, y, z] (nnode, 3)    
% GAP: gap field [1 x nStation]
%
% Solve placement problem
[data, U, GAP]=modelStationRigidPlacement(data, stationData, stationID);


