% Compute pre-load "F" using the current deviation field "U"
function data=modelSetPreLoad(data, Ka, U)
%
% Inputs
% data: data model
% Ka: assembly stiffness matrix (ndof x ndof) - sparse matrix
% U: solution vector (ndof x 1) - double matrix
%
% Outputs
% data: updated data model with the following fields:
    % .Assembly.PreLoad.F => [u, v, w, alfa, beta, gamma] force field 
    % .Assembly.PreLoad.DoF => [DoF IDs]
%
%--

if ~isfield(data.Input,'Part')
    return
end

if isempty(Ka) || isempty(U)
    return
end
%
nParts=length(data.Input.Part);
%
% Node ids
dofId=[];
for i=1:nParts
    if data.Input.Part(i).Enable && data.Input.Part(i).Status==0 % active and computed
        nodeIDi=data.Model.Nominal.Domain(i).Node;
        dofIdi=data.Model.Nominal.xMesh.Node.NodeIndex(nodeIDi,:)';
        dofId=[dofId, dofIdi(:)']; %#ok<AGROW>
    end
end
%
% Force vector
F = -Ka*U(dofId); % NOTE: reaction load
%
% save back
data.Assembly.PreLoad.F=[data.Assembly.PreLoad.F, F']; 
data.Assembly.PreLoad.DoF=[data.Assembly.PreLoad.DoF, dofId];

