% Build model
function data=modelBuild(data, optPart, optInput)

% optPart: 
    % "import" (default option):
        % import all geometry
        % refresh existing geometry
        % compute part UCS
        % do not compute stiffness matrix 
    % "refreshAll":
        % refresh existing geometry
        % recompute stiffness matrix
        % recompute part UCS
    % "refresh":
        % refresh existing geometry
        % do not update stiffness matrix
        % do not recompute part UCS
%--------------------------    
% optInput:    
    % "all" (default option):
        % refresh all
    % "features":
        % only refresh part features

if nargin==1
    optPart='import';
    optInput='all';
end
if nargin==2
    optInput='all';
end
%     
nsample=data.Assembly.X.nD;
for i=1:nsample
    data.Assembly.X.ID=i;
    data=modelBuildPart(data, optPart); 
    data=modelBuildInput(data, optInput);
end
%