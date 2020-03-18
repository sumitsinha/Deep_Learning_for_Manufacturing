% calculate closed boundaries of a given mesh
function [bmesh, flag]=getBoundaryMesh(fem)

% fem: fem structure
% bmesh: mesh boundary
% flag=0/1/2: not manifold mesh/closed mesh/open mesh...

%---
flag=0;

%---
bmesh=cell(0);

% get adjacency matrix
nvert=size(fem.xMesh.Node.Coordinate,1);

% get adjacency matrix
A=element2Adjacent(fem);

% check if any open boundary
BoundVertex=find(A==2); 

%--
if isempty(BoundVertex)
    boundEdge=[];
    flag=1; % closed mesh
    return
end

%--find seed
[boundEdge,A]=findSeed(A, nvert);

ActVertex=boundEdge(2); % actual vertex

% run loops
I=2;
CountLoop=1; % loop counter
CountVertex=1; % global counter
while(CountVertex<=nvert)
        
  ActEdge=find(A(ActVertex,:)==2); % connection to the previuos
  
    if length(ActEdge)~=2 % this implies not-manifold boundary
        boundEdge=[];
        flag=0; % not-manifold boundary
        break
    end
    
    % update A
    A(ActVertex,ActEdge)=0;
    
    if ActEdge(1)==boundEdge(I-1) % equal to previuos
        ActVertex=ActEdge(2);
    else
        ActVertex=ActEdge(1);
    end
    
    if ActVertex~=boundEdge(1)
        boundEdge=[boundEdge ActVertex];
        
        I=I+1; % 
    else % loop identified
        
        fprintf('   Boundary reconstruction: ID %g\n',CountLoop)
        
        boundEdge=[boundEdge ActVertex];
        bmesh{CountLoop}=boundEdge;
        flag=2; % at least one loop has been identified
        
        CountLoop=CountLoop+1; % update
        
        % get new seed
        [boundEdge,A]=findSeed(A, nvert);
        
        if isempty(boundEdge) % done
            break
        end
        
        %---
        ActVertex=boundEdge(2);
        
        I=2; % reset counter
    end

    CountVertex=CountVertex+1;
    
end

%-- find seed
function [boundEdge,A]=findSeed(A, nvert) 

% init
boundEdge=[]; 

%--
for I=1:nvert
    ActEdge=find(A(I,:)==2);
    if ~isempty(ActEdge)
        boundEdge=[I ActEdge(1)];
        
        %-- update
        A(I,ActEdge)=0;
        
        break; % stop when the first is identified
    end
end

