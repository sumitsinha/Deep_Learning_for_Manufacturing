% calculate free edges
function [edgeLoop, flag]=femFreeEdge(fem, idpart)
   
% fem: fem model

% boundPoints: boundary points' IDs
% flag=0/1/2: boundary not mainfold/closed mesh/open mesh

%--
flag=0;
edgeLoop=cell(0,1);

fem=femSubStructure(fem, fem.Domain(idpart).Element);

% get adjacency matrix
fprintf('    FEMP Boundary - calculating connections...\n')
A=element2Adjacent(fem, 'free edge');

% check for boundary vertex
nvert=size(A,1);
BoundVertex=find(A==1); 

%- closed mesh
if isempty(BoundVertex)
    flag=1; % closed mesh
    return
end

% get initial seed
[boundEdge,A]=findSeed(A);

ActVertex=boundEdge(2); 

% run loop
fprintf('    FEMP Boundary - calculating loops...\n')
I=2;
CountLoop=1; 
CountVertex=1; 
while(CountVertex<=nvert)
       
  ActEdge=find(A(ActVertex,:)==1); %actual vertex is connected to previuos
  
    if length(ActEdge)~=2 % non mainfold boundary
        flag=0; 
        break
    end
    
    %--
    A(ActVertex,ActEdge)=0;

    if ActEdge(1)==boundEdge(I-1) 
        ActVertex=ActEdge(2);
    else
        ActVertex=ActEdge(1);
    end
    
    if ActVertex~=boundEdge(1)
        boundEdge=[boundEdge ActVertex];
        
        I=I+1; 
    else % is actual vertex is equal to last, go to next loop
               
        boundEdge=[boundEdge ActVertex];
        edgeLoop{1,CountLoop}=fem.xMesh.Node.Coordinate(boundEdge,:);
        flag=2; % one loop found
        
        CountLoop=CountLoop+1; 
        
        % new seed
        [boundEdge,A]=findSeed(A);
        
        if isempty(boundEdge) % exit
            break
        end
        
        %--
        ActVertex=boundEdge(2);
        
        I=2; % reset counter
    end

    CountVertex=CountVertex+1;
    
end

%--
fprintf('    FEMP Boundary - %g loops founds\n', CountLoop-1);


%--
function [boundEdge,A]=findSeed(A) 

%--
boundEdge=[];

nvert=size(A,1);
for I=1:nvert
    ActEdge=find(A(I,:)==1);
    if ~isempty(ActEdge)
        boundEdge=[I ActEdge(1)];
        
        A(I,ActEdge)=0;
        
        break; 
    end
end

