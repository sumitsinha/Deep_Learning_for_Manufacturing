% import multiple mesh file
function fem=importMultiMesh(fem, datainput, opt)

% INPUT:
% fem: fem structure
% datainput: {filename} => opt='file'
% datainput: .Node; .Quad; .Tria => opt='geom'

% OUTPUT:
% fem: fem structure


if nargin==2
    opt='file';
end

[fem, log]=mesh2Fem(fem, datainput, opt);

% initialise selection
fem=femInitSelection(fem);

% calculate node 2 element connectivity
fem=femNode2Element(fem);

% calculate element 2 element connectivity
fem=femElement2Element(fem);

% setta gli UCS locali
nnode=size(fem.xMesh.Node.Coordinate,1);

fem.xMesh.Ucs=ones(1,nnode);
fem.xMesh.Reference=false(1,nnode);

fem.Geometry.Ucs{1}=eye(3,3);

disp('>>--')

nDom=fem.Sol.nDom;
if (nDom)>0
    fprintf('Mesh File Imported Successfully!\n')
else
    warning('No domain has been detected!')
end

fprintf('   Summary:\n')
fprintf('     no. of Domains: %g\n',nDom)
fprintf('     no. of Nodes: %g\n',log.nNodes)
fprintf('     no. of QUADs: %g\n',log.nQuads)
fprintf('     no. of TRIAs: %g\n',log.nTrias)

      




