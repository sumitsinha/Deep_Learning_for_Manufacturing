% save QUADs
function fem=storeQuads(fem, quad, kdom, idelement)

% get options
stiffbool=fem.Options.StiffnessUpdate;
massbool=fem.Options.MassUpdate; 

% QUAD
if quad(1)~=-1

    nquad=size(quad,1);
    
    fem.Sol.Quad.Count=fem.Sol.Quad.Count+nquad;
    
    fem.Sol.Quad.Id=[fem.Sol.Quad.Id idelement];
    
    fem.Sol.Quad.Element=[fem.Sol.Quad.Element
                          quad];
    
    if isempty(fem.xMesh.Element(1).Element)                  
        count=1;
    else
        count=length(fem.xMesh.Element)+1;
    end
    
    for i=1:nquad
        
        fem.xMesh.Element(count).Element=quad(i,:);
        fem.xMesh.Element(count).Type='quad';
        fem.xMesh.Element(count).Component=kdom;
        
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

        % stiffness
        if stiffbool
            fem.xMesh.Element(count).Ke=zeros(24,24); % stiffness matrix
        else
            fem.xMesh.Element(count).Ke=[];
        end
        
        % mass matrices
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
        
        count=count+1;
    end
    
end
