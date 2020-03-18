function edgeLoop=movingMeshBoundaryNodes(fem,...
                                                        A,...
                                                        idnode)

% INPUT
% fem: fem structure
% A: adjacency matrix
% idnode: node ids

% OUTPUT:
% edgeLoop.bnode: ids of boundary nodes
% edgeLoop.Nbnode: tangent vectors of boundary nodes
% edgeLoop.Pbnode: xyz of boundary nodes

edgeLoop=[];

nvert=size(A,1);

% initial seed
[boundEdge,A]=findSeed(A);

actVertex=boundEdge(2); % this is the actual vertex. The other vertices are linked to this one

% start search 
I=2;
CountLoop=1; % loop counter
CountVertex=1; % global cunter
while(CountVertex<=nvert)
    
    actEdge=find(A(actVertex,:)==1); %find next vertex
  
    % update A matrix
    A(actVertex,actEdge)=0;
    
    if actEdge(1)==boundEdge(I-1) % equal to previous
        actVertex=actEdge(2);
    else
        actVertex=actEdge(1);
    end
    
    if actVertex~=boundEdge(1) % grow-up current loop
        
        boundEdge=[boundEdge actVertex];
        
        I=I+1; % update counter of the single loop
        
    else % start new loop
        
        edgeLoop(CountLoop).bnode=boundEdge;
        
        CountLoop=CountLoop+1; % increment
        
        % find new seed
        [boundEdge,A]=findSeed(A);
        
        if isempty(boundEdge) % sexit iterations
            break
        end
        
        % ... actual vertex
        actVertex=boundEdge(2);
        
        I=2; % re-set local counter
    end

    CountVertex=CountVertex+1;
    
end

% get tangent vectors
nloop=length(edgeLoop);

for i=1:nloop
            
nnode=length(edgeLoop(i).bnode);

    edgeLoop(i).Nbnode=zeros(nnode,3);
    
    edgeLoop(i).Pbnode=fem.xMesh.Node.Coordinate(idnode(edgeLoop(i).bnode),:);
    
    for j=1:nnode
        
        if j==1
            id0=idnode(edgeLoop(i).bnode(j));
            id1=idnode(edgeLoop(i).bnode(end));
            id2=idnode(edgeLoop(i).bnode(j+1));
        elseif j==nnode
            id0=idnode(edgeLoop(i).bnode(j));
            id1=idnode(edgeLoop(i).bnode(j-1));
            id2=idnode(edgeLoop(i).bnode(1));
        else
            id0=idnode(edgeLoop(i).bnode(j));
            id1=idnode(edgeLoop(i).bnode(j-1));
            id2=idnode(edgeLoop(i).bnode(j+1));
        end
        
            P1=fem.xMesh.Node.Coordinate(id1,:);
            P2=fem.xMesh.Node.Coordinate(id2,:);

            P0=fem.xMesh.Node.Coordinate(id0,:);

            % related edge directions
            e1=P1-P0;
            e2=P2-P0;
            Ni=fem.xMesh.Node.Normal(id0,:);

            Nt1=cross(Ni,e1)/norm(cross(Ni,e1));
            Nt2=cross(e2,Ni)/norm(cross(e2,Ni));

            % save back
            Nti=mean([Nt1;Nt2],1);
            edgeLoop(i).Nbnode(j,:)=Nti/norm(Nti);
    end
    
end


% find seed vertex
function [boundEdge,A]=findSeed(A) 

% init
boundEdge=[]; 

nvert=size(A,1);

% loop over all vertices
for i=1:nvert
    actEdge=find(A(i,:)==1);
    
    if ~isempty(actEdge)
        boundEdge=[i actEdge(1)]; % take the first one
        
        % update A matrix
        A(i,actEdge)=0; % already visited
        break
    end
end

