function A=element2AdjacentTrias(element, nnode)

% INPUT
% element: element connection (3-node connection)
% nnode: no. of mesh nodes

% OUTPUT
% A: adjacency matrix: A(i,j)=0 if i and j are not connected; A(i,j)=alfa
% if i and j are connected. The degree of connectity if equal to "alfa"

nface=size(element,1);

% loop over all elements
id=[1 2
    1 3
    3 2];

% set initial values for sparse matrices
X=zeros(1,nface*6);
irow=zeros(1,nface*6);
jcol=zeros(1,nface*6);

count=1;
for i=1:nface
    face=element(i,:); 
    
    for j=1:3
        jj=id(j,1);
        zz=id(j,2);
        
            X(count)=1;
            irow(count)=face(jj);
            jcol(count)=face(zz);
            count=count+1;
            
            % symmetric condition ( edge (i,j) == edge (j,i) )
            X(count)=1;
            irow(count)=face(zz);
            jcol(count)=face(jj);
            count=count+1;
                        
    end
    
end

% fill sparse matrix
A=femSparse(irow,...
             jcol, ...
             X, ...
             nnode, nnode);




