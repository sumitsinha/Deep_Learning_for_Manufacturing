% map the solution "U" to the entire fem model
function Uout=mapSolutionAllDomain(fem, Uin)

%--
nnodet=size(fem.xMesh.Node.Coordinate,1);
nodeIndex=getIndexNode(1:nnodet, 6);

% set initial 
Uout=zeros(nnodet*6,1);

nnodea=fem.Selection.Node.Count;

% loop over active nodes
for i=1:nnodea
   
    % get current dofs
    id=fem.Selection.Node.Active(i);
    
    dofs=fem.xMesh.Node.NodeIndex(id,:);
    
    % get solution
    ui=Uin(dofs(:));
    
    % save solution
    dofs=nodeIndex(id,:);
    
    Uout(dofs(:))=ui;
    
end




