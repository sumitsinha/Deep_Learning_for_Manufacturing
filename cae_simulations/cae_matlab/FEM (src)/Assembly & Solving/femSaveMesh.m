function fem=femSaveMesh(fem, quad, tria, node)

% get options
stiffbool=fem.Options.StiffnessUpdate;
massbool=fem.Options.MassUpdate; 

% save NODE
nnode=size(node,1);

% nominal mesh
fem.xMesh.Node.Coordinate=node;
fem.xMesh.Node.Component=zeros(1,nnode);
fem.xMesh.Node.Normal=[zeros(nnode,2),ones(nnode,1)];
fem.xMesh.Node.NormalReset=[zeros(nnode,2),ones(nnode,1)];
fem.xMesh.Node.NodeIndex=zeros(nnode,6);

% deformed frame
fem.Sol.DeformedFrame.Node.Coordinate=node; 
fem.Sol.DeformedFrame.Node.Normal=[zeros(nnode,2),ones(nnode,1)]; 
fem.Sol.DeformedFrame.Node.NormalReset=[zeros(nnode,2),ones(nnode,1)]; 

fem.xMesh.Node.Tnode=cell(nnode,1);
for i=1:nnode
    fem.xMesh.Node.Tnode{i}=eye(3,3);
end

% save ELEMENTS
nquad=size(quad,1);
ntria=size(tria,1);

count=1;

% QUAD
if quad(1,1) ~= -1

    fem.Sol.Quad.Count=nquad;
    
    fem.Sol.Quad.Id=zeros(1,nquad);
    for i=1:nquad
        fem.xMesh.Element(count).Element=quad(i,:);
        fem.xMesh.Element(count).Type='quad';
        fem.xMesh.Element(count).Component=0;
        
        % mat. properties
        fem.xMesh.Element(count).Material.E=210e3; % Young
        fem.xMesh.Element(count).Material.ni=0.33; % Poisson
        fem.xMesh.Element(count).Material.lamda=5/6; % Shear correction factor
        fem.xMesh.Element(count).Material.Density=7900e-9; % density

        fem.xMesh.Element(count).Constant.Th=1; % thickness

        % matrices
        fem.xMesh.Element(count).ElementIndex=zeros(1,24);
        fem.xMesh.Element(count).ElementNodeIndex=zeros(4,6);
        
        fem.xMesh.Element(count).Tmatrix.T0lDoF=eye(24,24); 
        fem.xMesh.Element(count).Tmatrix.T0ucs=eye(24,24);
        
        fem.xMesh.Element(count).Tmatrix.T0lGeom=eye(3,3); % 3x3 rotation matrix
        fem.xMesh.Element(count).Tmatrix.P0lGeom=[0 0 0]; % origin of local coordinate frame
        
        fem.xMesh.Element(count).Tmatrix.Normal=[0 0 1];
        fem.xMesh.Element(count).Tmatrix.NormalReset=[0 0 1];

        % stiffness and mass matrices
        if stiffbool
          fem.xMesh.Element(count).Ke=zeros(24,24); % stiffness matrix
        else
          fem.xMesh.Element(count).Ke=[];
        end
        
        if massbool
         fem.xMesh.Element(count).Me=zeros(24,24); % mass matrix
        else
         fem.xMesh.Element(count).Me=[];
        end
        
        % deformed frame
        fem.Sol.DeformedFrame.Element(count).Tmatrix.T0lGeom=eye(3,3); % 3x3 rotation matrix
        fem.Sol.DeformedFrame.Element(count).Tmatrix.P0lGeom=[0 0 0]; % origin of local coordinate frame
        fem.Sol.DeformedFrame.Element(count).Tmatrix.Normal=[0 0 1];
        fem.Sol.DeformedFrame.Element(count).Tmatrix.NormalReset=[0 0 1];

        fem.Sol.Quad.Id(i)=count;
        
        count=count+1;
    end
    
    fem.Sol.Quad.Element=quad;
    
end

% TRIA
if tria(1,1)~=-1
    
    fem.Sol.Tria.Count=ntria;
    
    fem.Sol.Tria.Id=zeros(1,ntria);
    for i=1:ntria
        fem.xMesh.Element(count).Element=tria(i,:);
        fem.xMesh.Element(count).Type='tria';
        fem.xMesh.Element(count).Component=0;
        
        % mat. properties
        fem.xMesh.Element(count).Material.E=210e3; % Young
        fem.xMesh.Element(count).Material.ni=0.33; % Poisson
        fem.xMesh.Element(count).Material.lamda=5/6; % Shear correction factor
        fem.xMesh.Element(count).Material.Density=7900e-9; % density

        fem.xMesh.Element(count).Constant.Th=1; % thickness

        % matrices
        fem.xMesh.Element(count).ElementIndex=zeros(1,18);
        fem.xMesh.Element(count).ElementNodeIndex=zeros(3,6);
        
        fem.xMesh.Element(count).Tmatrix.T0lDoF=eye(18,18); 
        fem.xMesh.Element(count).Tmatrix.T0ucs=eye(18,18);
        
        fem.xMesh.Element(count).Tmatrix.T0lGeom=eye(3,3); % 3x3 rotation matrix
        fem.xMesh.Element(count).Tmatrix.P0lGeom=[0 0 0]; % origin of local coordinate frame
        
        fem.xMesh.Element(count).Tmatrix.Normal=[0 0 0];
        fem.xMesh.Element(count).Tmatrix.NormalReset=[0 0 0];

        % stiffness and mass matrices
        if stiffbool
          fem.xMesh.Element(count).Ke=zeros(18,18); % stiffness matrix
        else
          fem.xMesh.Element(count).Ke=[];
        end
        
        if massbool
         fem.xMesh.Element(count).Me=zeros(18,18); % mass matrix
        else
         fem.xMesh.Element(count).Me=[];
        end
        
        % deformed frame
        fem.Sol.DeformedFrame.Element(count).Tmatrix.T0lGeom=eye(3,3); % 3x3 rotation matrix
        fem.Sol.DeformedFrame.Element(count).Tmatrix.P0lGeom=[0 0 0]; % origin of local coordinate frame
        fem.Sol.DeformedFrame.Element(count).Tmatrix.Normal=[0 0 1];
        fem.Sol.DeformedFrame.Element(count).Tmatrix.NormalReset=[0 0 1];
        
        fem.Sol.Tria.Id(i)=count;
        
        count=count+1;
    end
    
    fem.Sol.Tria.Element=tria;

end

% initialise selection
fem=femInitSelection(fem);

disp('Mesh File Imported Successfully!')

% setta gli UCS locali
fem.xMesh.Ucs=ones(1,nnode);
fem.xMesh.Reference=false(1,nnode);

fem.Geometry.Ucs{1}=eye(3,3);

% calculate connected components
disp('Growing Components...')
fem=growingComponents(fem);
