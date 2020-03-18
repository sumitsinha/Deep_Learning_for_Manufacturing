% import mesh file
function [fem, iddom]=importMeshLight(fem, filename)

nf=length(filename);

iddom=cell(1, nf);

%--
ndom=0;
quad=[];
tria=[];
node=[];
for ii=1:nf
           
    [triai, quadi, nodei, flagi]=localreadmesh(filename{ii});
    
    if flagi

        nn=size(node,1);
        node=[node; nodei];
        
        if ~isempty(quadi)
             quad=[quad; quadi+nn];
        end
        
        if ~isempty(triai)
            tria=[tria; triai+nn];
        end

        %------------
        % save NODE
        nnode=size(nodei,1);

        % nominal mesh
        femi=femInit();

        femi.xMesh.Node.Coordinate=nodei;
        femi.xMesh.Node.Component=zeros(1,nnode);

        count=1;

        % QUAD
        if ~isempty(quadi)

            nquad=size(quadi,1);
            for i=1:nquad
                femi.xMesh.Element(count).Element=quadi(i,:);
                femi.xMesh.Element(count).Type='quad';
                femi.xMesh.Element(count).Component=0;

                count=count+1;
            end

        end

        % TRIA
        if ~isempty(triai)

            ntria=size(triai,1);
            for i=1:ntria
                femi.xMesh.Element(count).Element=triai(i,:);
                femi.xMesh.Element(count).Type='tria';
                femi.xMesh.Element(count).Component=0;

                count=count+1;
            end

        end
        
        c=ndom;
        femi=growingComponentsLight(femi);

        %--
        iddom{ii}=c+1:femi.Sol.nDom+c;
        
        ndom=ndom+femi.Sol.nDom;
    
    end

end

%--
%------------
% save NODE
nnode=size(node,1);

% nominal mesh
fem.xMesh.Node.Coordinate=node;
fem.xMesh.Node.Component=zeros(1,nnode);

count=1;

% QUAD
if ~isempty(quad)

    nquad=size(quad,1);
    for i=1:nquad
        fem.xMesh.Element(count).Element=quad(i,:);
        fem.xMesh.Element(count).Type='quad';
        fem.xMesh.Element(count).Component=0;

        count=count+1;
    end

end

% TRIA
if ~isempty(tria)

    ntria=size(tria,1);
    for i=1:ntria
        fem.xMesh.Element(count).Element=tria(i,:);
        fem.xMesh.Element(count).Type='tria';
        fem.xMesh.Element(count).Component=0;

        count=count+1;
    end

end

% grow domain
fem=growingComponentsLight(fem);

% save normals
fem=localSetNormalVector(fem);

%-----------------
function [tria, quad, node, flag]=localreadmesh(filename)

flag=true;
tria=[];
quad=[];
node=[];

if isempty(filename)
    flag=false;
    return
end

% get file extension
[~,~,ext]=fileparts(filename);

% abaqus
if strcmp(ext,'.inp')
    disp('Reading Abaqus Mesh File...')
    [quad,tria,node]=readMeshAbaqus(filename); % # MEX function
    
% nastran
elseif strcmp(ext,'.bdf') || strcmp(ext,'.dat')
    disp('Reading Nastran Mesh File...')
    [quad,tria,node]=readMeshNas(filename); % # MEX function
    
% stl format
elseif strcmp(ext,'.stl') || strcmp(ext,'.STL') 
    disp('Reading STL Mesh File...')
    
    stltype = getStlFormat(filename);
    
    if strcmp(stltype,'ascii')
        
        [tria,node]=readMeshStlAscii(filename); % matlab function
        
    elseif strcmp(stltype,'binary')
        
        [tria, node]=readMeshStlBin(filename); % matlab function
        
    end
    
else
    flag=false;
    disp('Mesh file not reconognised')
    return
end

% normal vector
function fem=localSetNormalVector(fem)

n=size(fem.xMesh.Node.Coordinate,1);
fem.xMesh.Node.Normal=zeros(n,3);
for i=1:n
    
    idele=fem.Sol.Node2Element{i};
    
    nele=length(idele);
    Nn=[0 0 0];
    for j=1:nele
          Ni=localNormal(fem.xMesh.Element(idele(j)).Element, fem.xMesh.Node.Coordinate);
          Nn=Nn+Ni;   
    end
    
    %get normal vector by average method
    Nn=Nn/nele;
    Nn=Nn/norm(Nn);
    
    % save normal
    fem.xMesh.Node.Normal(i,:)=Nn;
    
end

%---
function normf=localNormal(facei, vertexi)

nf=size(facei,1);
normf=zeros(nf,3);

for i=1:nf
    pi=vertexi(facei(i,:), :);

    p1=pi(1,:);
    p2=pi(2,:);
    p3=pi(3,:);

    ni=cross(p2-p1, p3-p1);
    normf(i,:)=ni/norm(ni);
end


% apply a growing procedure to group connected components
function fem=growingComponentsLight(fem)

% calculate node 2 element connectivity
fem=femNode2Element(fem);

% calculate element 2 element connectivity
fem=femElement2Element(fem);

nDom=1;
while true
        
    % get initial seed
    [iseed, fem]=getParentNode(fem, nDom);
    
    idparent=iseed;
    
    if isempty(idparent)
        break
    end
    
    st=sprintf('... growing Component ID: %d', nDom);
    disp(st)
   
    % connectivity
    fem.Domain(nDom).Element=[];
    fem.Domain(nDom).ElementTria=[];
    fem.Domain(nDom).ElementQuad=[];
    
    fem.Domain(nDom).Node=[];
    
    % grow domain "nDom" locally
    while true
        
        nparent=length(idparent);
        idEle=[];
        for i=1:nparent
          [temp, fem]=getNode2Element(fem, idparent(i), nDom);
          
          idEle=[idEle,temp];
        end
        
        % check if a new domain has been filled
        if isempty(idEle)
            
            fem.Domain(nDom).Node(end+1)=iseed;
            
            nDom=nDom+1;

            break
        end
    
        nElec=length(idEle);
        idparent=[];
        for i=1:nElec
         [temp, fem]=getChildrenNode(fem, idEle(i), nDom);
         
         idparent=[idparent,temp];
        end
        
    end
    
end

%-- save domain number
fem.Sol.nDom=nDom-1;


% get seed Node
function [idseed, fem]=getParentNode(fem, nDom)

idseed=[];

nnode=size(fem.xMesh.Node.Coordinate,1);

for i=1:nnode
    if fem.xMesh.Node.Component(i)==0 % never visited
        idseed=i;
        
        fem.xMesh.Node.Component(i)=nDom; % now this node is taged as visited
        
        return
    end
end

% get child nodes
function [idchild, fem]=getChildrenNode(fem, idele, nDom)

idchild=[];

temp=fem.xMesh.Element(idele).Element;

n=length(temp);
for i=1:n
    
    if fem.xMesh.Node.Component(temp(i))==0 % never visited
        idchild=[idchild,temp(i)];
        
        fem.xMesh.Node.Component(temp(i))=nDom;  % now this element is taged as visited
        
        fem.Domain(nDom).Node=[fem.Domain(nDom).Node,temp(i)];   
    end
        
end


% get elements connected to parent
function [idEle, fem]=getNode2Element(fem, idnode, nDom)

idEle=[];

temp=fem.Sol.Node2Element{idnode};

n=length(temp);
for i=1:n
    
    % element type
    etype=fem.xMesh.Element(temp(i)).Type;
    
    if fem.xMesh.Element(temp(i)).Component==0 % never visited
        idEle=[idEle, temp(i)];
        
        fem.xMesh.Element(temp(i)).Component=nDom;  % now this element is taged as visited
        
        fem.Domain(nDom).Element=[fem.Domain(nDom).Element,temp(i)];    
        
         %---
         if strcmp(etype,'tria')
             fem.Domain(nDom).ElementTria=[fem.Domain(nDom).ElementTria,temp(i)];
         elseif strcmp(etype,'quad')
             fem.Domain(nDom).ElementQuad=[fem.Domain(nDom).ElementQuad,temp(i)];
         end
             
        
    end
    
end

