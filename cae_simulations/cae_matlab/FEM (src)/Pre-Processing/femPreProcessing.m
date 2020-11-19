% pre-process FEM model of selected "activeNode"
function fem=femPreProcessing(fem, activeNode)

% fem: fem structure
% activeNode(id).Node: list of nodes to be activated
% activeNode(id).Status: -1/0 NOT active/active
% activeNode(id).Part: part IDs

% If "activeNode" is empty then use all nodes of the given parts

%--
% calculate:
% 1: dofs indexes
% 2: rotation matrices/normal vectors
% 3: UCS matrices
% 4: stiffness matrices
% 5: no. of dofs
% 6: data for mesh denoising

if nargin==1
    activeNode=[];
end

%-------------
disp('Setting Domain Properties...') 
fem=setDomainProperties(fem);

% set active domains
disp('Setting Active Domains...') 
fem=femSetSelection(fem, activeNode);

disp('Calculating Element Matrices...') 
disp('   set initial values...') 

nele=length(fem.xMesh.Element);

for i=1:nele
    
    etype=fem.xMesh.Element(i).Type;
    
    if strcmp(etype,'tria')
        
            % re-set stiffness matrices
            if fem.Options.StiffnessUpdate
                fem.xMesh.Element(i).Ke=zeros(18,18);
            end 

            % re-set mass matrix
            if fem.Options.MassUpdate
                fem.xMesh.Element(i).Me=zeros(18,18);
            end 
            
    elseif strcmp(etype,'quad')
        
            % re-set  stiffness matrices
            if fem.Options.StiffnessUpdate
                fem.xMesh.Element(i).Ke=zeros(24,24);
            end 

            % re-set mass matrix
            if fem.Options.MassUpdate
                fem.xMesh.Element(i).Me=zeros(24,24);
            end 
    
    end

end
    

disp('Calculating Element Matrices...') 
fem=femPreProcessingMatrix(fem);

%----------------------------------------
% update deformated frame
disp('Initialising Deformed Frame...') 

fem.Sol.DeformedFrame.Node.Coordinate=fem.xMesh.Node.Coordinate;
fem.Sol.DeformedFrame.Node.Normal=fem.xMesh.Node.Normal;
fem.Sol.DeformedFrame.Node.NormalReset=fem.xMesh.Node.NormalReset;

% update elements
for i=1:nele
    fem.Sol.DeformedFrame.Element(i).Tmatrix=fem.xMesh.Element(i).Tmatrix;
end

%----------------------------------------
% update the list of node indices
if fem.Options.UseActiveSelection
    nnode=length(fem.Selection.Node.Active);    
    
    fem.xMesh.Node.NodeIndex(fem.Selection.Node.Active,:)=getIndexNode(1:nnode, 6);
else
    nnode=size(fem.xMesh.Node.Coordinate,1);
    fem.xMesh.Node.NodeIndex=getIndexNode(1:nnode, 6);
end

% save not-zero-numbers after getting stiffness items
%----------------------------------------------------
fem.Sol.Kast.ndofs=fem.Sol.Kast.n;
%----------------------------------------------------

% calculate normal vector for each node...
disp('Calculating Node Normal Vectors...')
fem=getNodeNormalReferenceFrame(fem);

% calculate data for mesh denoising:
idparts=1:fem.Sol.nDom;
fem=femmesh2triamesh(fem, idparts);

%--
disp('Pre-processing Completed!') 
