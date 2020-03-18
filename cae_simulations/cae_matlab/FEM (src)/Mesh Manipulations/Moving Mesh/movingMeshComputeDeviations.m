function [fem, u, v, w]=movingMeshComputeDeviations(fem,...
                                                     idpart,...
                                                     cloud,...
                                                     niter,...
                                                     dsearchn)

% INPUT
% fem: fem structure
% ipart: id domain to be work on
% cloud: xyz coordinates of CoP
% niter: max. no. of iterations
% dsearch: searching distance for deviation calculation

% OUTPUT
% fem: updated fem structure
% u/v/w: deviations along x, y, z for each mesh node (entire mesh model!)

disp('>>--')

%----
% get initial xyz node coordinates
nodei=fem.xMesh.Node.Coordinate;

% get element and node ids
idtria=fem.Denoise.Domain(idpart).Tria;
idnode=fem.Domain(idpart).Node; 

% element connection
element=fem.Denoise.Tria(idtria,:);

% renumber element (-- for safe code --)
element=renumberMesh(element, idnode);

% compute laplacian matrix
fprintf('Compute mesh laplacian\n');
%-------------------------------

nnode=length(idnode); % no. of nodes
L=movingMeshComputeMeshWeight(element, nnode);
%-------------------------------

subs=0.05;

% run iterations
for k=1:niter
    
    fprintf('      Moving Mesh: iteration %g\n',k);
    
    node=fem.xMesh.Node.Coordinate(idnode,:); % xyz
                
    fprintf('      Moving Mesh: select anchor point\n');
    
    % constraint assigment
    
    % get constraint directions          
    Nanchor=fem.xMesh.Node.Normal(idnode,:); % normal vectors
    Panchor=node;

    idanchor=1:size(Panchor,1); 
    
    %%
%     n=size(Panchor,1);
%     idanchor=randperm(n);
%     
%     %idanchor=idanchor(1:floor(n*subs));
%         
%     idanchor=idanchor(1:20);
   
    %--
    
    %... then,
    Panchor=Panchor(idanchor,:); % xyz coordinates of anchor points
    Nanchor=Nanchor(idanchor,:); % normal vectors for anchor points
    
    % get deviations
    dev=getNormalDevPoints2Points(Panchor, Nanchor, cloud, dsearchn);
       
    % remove points with "0" deviation (these might correspond to missing regions in the CoP)
    Panchor(dev==0,:)=[];
    Nanchor(dev==0,:)=[];

    idanchor(dev==0)=[];

    dev(dev==0)=[];
    %-------------------------------
    
    % assembly equations
    fprintf('      Moving Mesh: get constraint equations\n');
    [C, q]=movingMeshConstraintEquations(Panchor, Nanchor, idanchor, dev, nnode);

    % solve equations
    fprintf('      Moving Mesh: solve equations (week-form)\n');
    node=movingMeshSolveEquations(L, C, q, node);

    % save back and update
    fem.xMesh.Node.Coordinate(idnode,:)=node;
    
    %--
    fprintf('      Moving Mesh: update FEM model\n');
    fem.Options.StiffnessUpdate=false; % stiffness matrix not updated
    fem.Options.ConnectivityUpdate=false; % connectivity matrices not updated
    fem=femPreProcessing(fem);

end

% get final xyz node coordinates
nodee=fem.xMesh.Node.Coordinate;

% get final deviations

% u:
u=nodee(:,1)-nodei(:,1);

% v:
v=nodee(:,2)-nodei(:,2);

% w:
w=nodee(:,3)-nodei(:,3);

disp('>>--')

