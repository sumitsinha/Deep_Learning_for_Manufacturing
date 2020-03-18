%-------------------------------
%-------------------------------
function A=element2Adjacent(fem, opt)

% INPUT
% fem:fem structure
% opt:
    % "selection":
    % "free edge":

% OUTPUT
% A: adjacency matrix: A(i,j)=0 if i and j are not connected; A(i,j)=alfa
% if i and j are connected. The degree of connectity if equal to "alfa"

%------------
% N.B: it finds out the nodes between selected and un-selected regions
%------------

if nargin==1
    opt='selection';
end

nele=length(fem.xMesh.Element);
nnode=size(fem.xMesh.Node.Coordinate,1);

ntria=fem.Sol.Tria.Count;
nquad=fem.Sol.Quad.Count;

xrow=zeros(1, ntria*6 + nquad*8);
xcol=zeros(1, ntria*6 + nquad*8);
X=zeros(1, ntria*6 + nquad*8);
c=1;

% loop over all elements
for i=1:nele
    
    etype=fem.xMesh.Element(i).Type;
    element=fem.xMesh.Element(i).Element;
    
    if strcmp(opt, 'selection') % selection mode
        boolele=fem.Selection.Element.Status(i);
    end
    
    if strcmp(etype,'quad')
        
        edge=[element(1) element(2)
              element(2) element(3)
              element(3) element(4)
              element(1) element(4)];
          
    elseif strcmp(etype,'tria')
        
        edge=[element(1) element(2)
              element(2) element(3)
              element(1) element(3)];
          
    end
    
   nedge=size(edge,1);
   
   %  loop over all edges
   for j=1:nedge     
       
           jj=edge(j,1);
           zz=edge(j,2);
           
           if strcmp(opt, 'selection') % selection mode
               
                   if boolele % active selection

                       xrow(c)=jj;
                       xcol(c)=zz;
                       X(c)=2;

                       c=c+1;

                       % symmetric condition ( edge (i,j) == edge (j,i) )
                       xrow(c)=zz;
                       xcol(c)=jj;
                       X(c)=2;

                       c=c+1;

                   else % not active selection

                       xrow(c)=jj;
                       xcol(c)=zz;
                       X(c)=-1;

                       c=c+1;

                       % symmetric condition ( edge (i,j) == edge (j,i) )
                       xrow(c)=zz;
                       xcol(c)=jj;
                       X(c)=-1;

                       c=c+1;

                   end
           else % free edge
               
                   xrow(c)=jj;
                   xcol(c)=zz;
                   X(c)=1;

                   c=c+1;

                   % symmetric condition ( edge (i,j) == edge (j,i) )
                   xrow(c)=zz;
                   xcol(c)=jj;
                   X(c)=1;

                   c=c+1;
                           
           end
           
   end
    
end

% assembly of A
A=femSparse(xrow, xcol, X,...
            nnode,...
            nnode); 


