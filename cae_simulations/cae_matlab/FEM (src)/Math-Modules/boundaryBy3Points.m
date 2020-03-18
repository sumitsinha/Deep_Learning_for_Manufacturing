% Calculate boundary node ids which falls into the segment defined by the "pselection"
function [boundaryIds, flag]=boundaryBy3Points(fem, pselection, sdist)

% fem: fem model
% pselection: xyz of the 3 points on (or close) to the free edge 
    % P1: start point (xyz)
    % P2: ending point (xyz)
    % P3: via point(xyz)
% sdist: searching distance
    
% boundaryIds: boundary points' IDs
% flag:
    % 0 correct identification
    % 1 boundary not-identified
    % 2 error with mesh connectivity
    % 3 no point detected in the searching region

% Init outputs args
flag=0;
boundaryIds=[];

%--
% get adjacency matrix
fprintf('    FEMP Boundary - calculating connections...\n')
A=element2Adjacent(fem, 'free edge');

% get ids of free edge nodes
idfreeedges=lncGetFreeEdges(A);
if isempty(idfreeedges)
    flag=false;
    return
end

% Retrieve node ids of the "freeedges" which are closest to "pselection"
[idfreeedgeselect, flagt, ~]=xyz2Id(fem, pselection(1:2,:), idfreeedges, sdist); 

if ~flagt
    flag=3; % no points detected in the searching region
    return
end

% initial seed
actEdge=find(A(idfreeedgeselect(1),:)==1);

if isempty(actEdge)
    flag=1; % boundary not-identified
    return
end

nloops=length(actEdge); % no. of possible detectable boundaries
if nloops~=2
    flag=2; % error with mesh connectivity
    return
end

% Run over loops
Aback=A; % create backup of A matrix
loopsBound=cell(1,nloops);
mdist=inf;
bmin=0;
for k=1:nloops
     A=Aback;
     tboundPoints=[idfreeedgeselect(1) actEdge(k)]; 
     tboundPoints=lclSearch(idfreeedgeselect, A, tboundPoints);
     
     loopsBound{k}=tboundPoints;
     
     % Compute distance of boundary loop to via point "pselection(3,:)"
     mdistk=distanceCheckPoint2Loop(fem, tboundPoints, pselection(3,:));

                %      % compute the closest distance from "pselection(3,:)" of the just computed lopp
                %      [~, ~, mdistk]=xyz2Id(fem, pselection(3,:), tboundPoints, inf);
     
     if mdistk<=mdist
         mdist=mdistk; % update current min loop
         bmin=k;
     end

end
     
% save back
boundaryIds=loopsBound{bmin};
     

% Compute distance of boundary loop to checking point
function dmin=distanceCheckPoint2Loop(fem, loopsBound, pcheck)

% fem: fem model
% loopsBound: sequence of boundary ids 
% pcheck: xyz of point to be checked

ploopsBound=fem.xMesh.Node.Coordinate(loopsBound, :);

% Enrich boundary point (add 1 middle point for each segement)
nnode=size(ploopsBound,1);
ploopsBoundE=zeros(2*nnode-1,3);
ploopsBoundE(1:nnode,:)=ploopsBound;
for i=1:nnode-1
    pi=ploopsBound(i,:);
    pj=ploopsBound(i+1,:);
    
    vij=pj-pi;
    pm=pi+vij*0.5; % middle point
    
    ploopsBoundE(nnode+i,:)=pm;
end

% compute min distance
d=zeros(2*nnode-1,3);
d(:,1)=ploopsBoundE(:,1)-pcheck(1);
d(:,2)=ploopsBoundE(:,2)-pcheck(2);
d(:,3)=ploopsBoundE(:,3)-pcheck(3);

di=sqrt(sum(d.^2, 2));

dmin=min(di);


%---------------------------------------
% run local search
function boundEdge=lclSearch(pb, A, boundEdge)

% pb: 
    % P1: start point ID
    % P2: ending point ID
    % P3: via point ID
    
%----------
actVertex=boundEdge(2); % this is the actual vertex. The other vertices are linked to this one
if actVertex==pb(2) % grow-up current loop
    return
end

nnode=size(A,1);

% start search 
I=2;
CountVertex=1;
while(CountVertex<=nnode)
    
    actEdge=find(A(actVertex,:)==1); %find next vertex
    
    if isempty(actEdge)
        break
    end
  
    % update A matrix
    A(actVertex,actEdge)=0;
    
    if actEdge(1)==boundEdge(I-1) % equal to previous
        actVertex=actEdge(2);
    else
        actVertex=actEdge(1);
    end
    
    if actVertex~=pb(2) % grow-up current loop
        
        boundEdge=[boundEdge actVertex];
        
        I=I+1; % update counter of the single loop
        
    else % save boundary and exit
       
      boundEdge=[boundEdge pb(2)]; 
      break
      
    end
    
    CountVertex=CountVertex+1;
    
end


% Retrieve node ids of the "freeedges" which are closest to "pselection"
function [idfreeedgeselect, flag, dmin]=xyz2Id(fem, pselection, idfreeedges, sdist)

% fem: fem model
% pselection: xyz of the selection
% idfreeedges: ids of free edges
% sdist: searching distance

% idfreeedgeselect: id of closest free edges
% flag: true/false: within "sdist / outside of "sdist"
% dmin: min distance of pselection to free edges

% init...
flag=true;
dmin=0;

% xyz of free edges
pfreeedges=fem.xMesh.Node.Coordinate(idfreeedges, :);
nnode=size(pfreeedges,1);

% no. of searching points
np=size(pselection,1);

idfreeedgeselect=zeros(np,1);
for i=1:np
    d=zeros(nnode,3);
    d(:,1)=pfreeedges(:,1)-pselection(i,1);
    d(:,2)=pfreeedges(:,2)-pselection(i,2);
    d(:,3)=pfreeedges(:,3)-pselection(i,3);

    di=sqrt(sum(d.^2, 2));

    [dmin, idi]=min(di);
    if dmin<=sdist % check distance
        idfreeedgeselect(i)=idfreeedges(idi);
    else
        flag=false;
        break
    end
end

% get ids of boundary nodes
function bnode=lncGetFreeEdges(A)

% INPUT
% A: laplace matrix

% OUTPUT
% bnode: list of boundary nodes ids

% loop over all nodes
[bnode, ~]=find(A==1);

% remove duplicates
bnode=unique(bnode);

