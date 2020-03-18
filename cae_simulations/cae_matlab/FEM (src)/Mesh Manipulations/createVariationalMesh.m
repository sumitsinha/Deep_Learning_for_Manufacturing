% create variational geometry based on "dev" vector
function fem=createVariationalMesh(fem,...
                                       dev)

% 
fem.xMesh.Node.Coordinate=fem.xMesh.Node.Coordinate + fem.xMesh.Node.Normal.*repmat(dev,1,3);
