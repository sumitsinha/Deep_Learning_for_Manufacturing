function A=edge2Adjacent(edge)

% INPUT
% edge: edge matrix (ne, 2)

% OUTPUT
% A: adjacency matrix: A(i,j)=0 if i and j are not connected; A(i,j)=1 if i and j are connected

ne=size(edge,1);
nnode=max(edge(:));

% set initial values for sparse matrices
X=zeros(1,ne*2);
irow=zeros(1,ne*2);
jcol=zeros(1,ne*2);

count=1;
for i=1:ne
    edgei=edge(i,:); 
    
    X(count)=1;
    irow(count)=edgei(1);
    jcol(count)=edgei(2);
    count=count+1;    
    
    % store symmatric part
    X(count)=1;
    irow(count)=edgei(2);
    jcol(count)=edgei(1);
    count=count+1;   
end

% fill sparse matrix
A=femSparse(irow,...
             jcol, ...
             X, ...
             nnode, nnode);
         
         