% calculate node to element connectivity
function fem=femNode2Element(fem)

% no. of elements and nodes
nele=length(fem.xMesh.Element);

nnode=size(fem.xMesh.Node.Coordinate,1);

% set initial values
fem.Sol.Node2Element=cell(1,nnode);

for i=1:nele
    ni=length(fem.xMesh.Element(i).Element); % no. of node per element
    
    for k=1:ni
        fem.Sol.Node2Element{fem.xMesh.Element(i).Element(k)}(end+1)=i; % fill
    end
end