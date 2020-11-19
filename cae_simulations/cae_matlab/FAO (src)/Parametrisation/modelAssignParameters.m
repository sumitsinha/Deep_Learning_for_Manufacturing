% Assign model parameters
function data=modelAssignParameters(data, nStation)

% Inputs:
% data: data structure with the following fields
    % .Assembly.X.ID: sample ID to be assigned
% nStation: no. stations

% Outputs:
% data: updated data structure

%
% check input
dv=data.Assembly.Parameter;
if isempty(dv)
    warning('Assigning parameters (error): no parameter detected - check input!');
    return
end
%
if isempty(data.Assembly.X.Value)
    warning('Assigning parameters (error): no sample detected - check input!');
    return
end
%
% STEP 1: Assign parameters to model
% Reset placement
if ~isfield(data.Input, 'Part')
    error('Assigning parameters (error): no part has been defined!')
end
%
npart=length(data.Input.Part);
for partID=1:npart
    data.Input.Part(partID).Placement.T=eye(4,4); % Inizialise the placement matrix
    data.Input.Part(partID).Placement.UCS=data.Input.Part(partID).Placement.UCSreset;
    %--
    data.Input.Part(partID).Placement.TStore=cell(1,nStation);
    for stationID=1:nStation
        data.Input.Part(partID).Placement.TStore{stationID}=eye(4,4);
    end
end
%
npara=size(dv,1);
sampleID=data.Assembly.X.ID;
for i=1:npara
    partID=dv{i,3};
    if dv{i,1}==0 % @"Non-ideal part"
        pointID=dv{i,4};
        data.Input.Part(partID).Morphing(pointID).DeltaPc=data.Assembly.X.Value(sampleID, i);
    elseif dv{i,1}==1 % @"Placement"
        T0w=data.Input.Part(partID).Placement.TStore{dv{i,5}};
        data.Input.Part(partID).Placement.UCS=data.Input.Part(partID).Placement.UCSStore{dv{i,5}};
        T0w=modelSetParametersPlacement(data,...
                                          data.Assembly.X.Value(sampleID, i),...
                                          dv{i,3},...
                                          dv{i,4},...
                                          dv{i,6},...
                                          T0w);
        data.Input.Part(partID).Placement.TStore{dv{i,5}}=T0w;
    elseif dv{i,1}==2 % @"Position"
        data=modelSetParametersPosition(data,...
                                            data.Assembly.X.Value(sampleID, i),...
                                            dv{i,3},...
                                            dv{i,4},...
                                            dv{i,5},...
                                            dv{i,6},...
                                            dv{i,7});
    end
end
%