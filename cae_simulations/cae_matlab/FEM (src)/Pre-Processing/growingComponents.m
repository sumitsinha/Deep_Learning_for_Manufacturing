% apply a growing procedure to group connected components
function fem=growingComponents(fem)

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
    
    % set default fields
    fem.Domain(nDom).Status=true; % by default all elements of this domain are active
    
    % connectivity
    fem.Domain(nDom).Element=[];
    fem.Domain(nDom).ElementTria=[];
    fem.Domain(nDom).ElementQuad=[];
    
    fem.Domain(nDom).Node=[];
    
    % material and constants (initial values)
    fem.Domain(nDom).Material.E=210e3;
    fem.Domain(nDom).Material.ni=0.33; 
    fem.Domain(nDom).Material.lamda=5/6; 
    fem.Domain(nDom).Material.Density=1; 
    
    fem.Domain(nDom).Constant.Th=1; 
    
    fem.Domain(nDom).NormalFlip=false;

    % domain load
    fem.Domain(nDom).Load.Value=[0 0 0 0 0 0]; % zero by default
    fem.Domain(nDom).Load.Flag=false; % false by default
    fem.Domain(nDom).Load.Frame='ref';
        
    % Sub-modelling options
    %--------------------------------------------------------------------------

    fem.Domain(nDom).SubModel.CuttingSt=[1 2 3]; 
    fem.Domain(nDom).SubModel.CuttingId=[1 2 3]; 
    fem.Domain(nDom).SubModel.SearchDist=10; 
    
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
    
    