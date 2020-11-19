% Solve multi-station assembly simulation
function [U, GAP, flagSimulation, inputData]=modelStationSolve(data, stationData)

% Inputs:
% data: model
% stationData: station data
        % "Non Ideal Part" => non ideal part
        % "Rigid Placement" => placement using roto-traslation of parts
        % "Locator Placement" => placement using hole and slot (if they have been defined)
        % "Clamp" => clamping/fastening
        % "Release" => release

% Outputs:
% inputData: updated inputData field =[1xnStation]   
% U: deviation fields [nDoF, nStation]
    % Cumulative deviation at station[i]=sum(U(1 to i))
% GAP: gap field [1 x nStation]
% flagSimulation: simulation flag [1 x nStation]
% inputData: input data fields [1 x nStation]
%----------------

%--
nStation=length(stationData);
%--
%
% Initialise
U=zeros(data.Model.Nominal.Sol.nDoF,nStation);
GAP=cell(1,nStation);
flagSimulation=false(1,nStation);
inputData=cell(1,nStation);
%
% Loop over all stations
Fai=[];
Ui=[];
for stationID=1:nStation
    %
    Gapi=[];
    flagSimulationi=false;
    %
    stationType=stationData(stationID).Type{1};
    
    % STEP 1 - set station
    data=modelStationSet(data, stationData, stationID);
    
    % STEP 2 - solve station
    if stationType==0 % "Non-ideal parts"
        [data, Ui, Gapi]=modelStationNonidealParts(data, stationData, stationID);
        flagSimulationi=true;
    elseif stationType==1 % "Rigid Placement"
        [data, Ui, Gapi]=modelStationRigidPlacement(data, stationData, stationID);
        flagSimulationi=true;
    elseif stationType==2 % "Locator Placement"
        [data, Ui, Gapi]=modelStationLocatorPlacement(data, stationData, stationID);
        flagSimulationi=true;
    elseif stationType==3 || stationType==4 % "Clamp" or "Release"
        % (1) Set pre-load
        data=modelSetPreLoad(data, Fai);
        if stationType==3
            data.Assembly.Solver.UsePreLoad=false;
        elseif stationType==4
            data.Assembly.Solver.UsePreLoad=true;
        end
        % (2) Solve
        [data, Ui, Gapi, Fai, flagSimulationi]=modelStationClamp(data, stationID);    
        % (3) Set "new" nominal (deformed geometry) & apply deformation field to part features
        data=modelSetGeometryToDeformed(data, stationData, stationID, 1, Ui);   
        % (4) Release pre-load
        if stationType==4 
            data.Assembly.PreLoad=[];
            data.Assembly.PreLoad(1).Value=[];
            data.Assembly.PreLoad(1).Domain=[];
            Fai=[];
        end
    else
        
        
        
        %----------------------------
        % Add here any other station type
        %----------------------------
        
        
        
    end
    
    % STEP 3: Update results
    if ~isempty(Ui)
        U(:,stationID)=Ui;
        Ui=[];
    end
    GAP{stationID}=Gapi;
    flagSimulation(stationID)=flagSimulationi;
    inputData{stationID}=data.Input;
            
    % STEP 4: Reset station & update part features
    data=modelStationReset(data);

    % STEP 5: Set to nominal all parts of the current station
    for i=stationData(stationID).Part
        data.Input.Part(i).Geometry.Type{1}=1; % nominal
    end
    
end

