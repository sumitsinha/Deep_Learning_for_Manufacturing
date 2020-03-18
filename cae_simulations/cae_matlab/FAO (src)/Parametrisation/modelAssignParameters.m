% Assign model parameters
function data=modelAssignParameters(data)

% Inputs:
% data: data structure with the following fields
    % .Assembly.X.ID: sample ID to be assigned

% Outputs:
% data: updated data structure

%
% check input
dv=data.Assembly.Parameter;
if isempty(dv)
    error('Assigning parameters (error): no parameter detected - check input!');
end
%
if isempty(data.Assembly.X.Value)
    error('Assigning parameters (error): no sample detected - check input!');
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
end
%
npara=size(dv,1);
sampleID=data.Assembly.X.ID;
for i=1:npara
    partID=dv{i,3};
    if dv{i,1}==1 % @"Placement"
        T0w=data.Input.Part(partID).Placement.T;
        T0w=modelSetParametersPlacement(data,...
                                          data.Assembly.X.Value(sampleID, i),...
                                          dv{i,3},...
                                          dv{i,4},...
                                          dv{i,6},...
                                          T0w);
        data.Input.Part(partID).Placement.T=T0w;
    end
    %--
    if dv{i,1}==2 % @"Position"
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