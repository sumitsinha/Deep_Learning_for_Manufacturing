% Compute pre-load "F" using the current deviation field "U"
function data=modelSetPreLoad(data, Fa)
%
% Inputs
% data: data model
% Fa: reaction forces
%
% Outputs
% data: updated data model with the following fields:
    % .Assembly.PreLoad.F => [u, v, w, alfa, beta, gamma] force field 
    % .Assembly.PreLoad.Fomain => list of domains
%
%--

if ~isfield(data.Input,'Part')
    return
end

if isempty(Fa) 
    return
end
%
nParts=length(data.Input.Part);
%
% Node ids
idpart=[];
for i=1:nParts
    if data.Input.Part(i).Enable && data.Input.Part(i).Status==0 % active and computed        
        idpart=[idpart, i]; %#ok<AGROW>
    end
end
%
% save back
c=length(data.Assembly.PreLoad);
c=c+1;
data.Assembly.PreLoad(c).Value=-Fa'; % reaction force
data.Assembly.PreLoad(c).Domain=idpart;

