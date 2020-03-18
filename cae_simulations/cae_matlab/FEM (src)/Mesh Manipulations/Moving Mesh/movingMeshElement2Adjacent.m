function A=movingMeshElement2Adjacent(element, nnode)

% INPUT
% element: element connection (3-node connection)
% nnode: no. of mesh nodes

% OUTPUT
% A: adjacency matrix: A(i,j)=0 if i and j are not connected; A(i,j)=alfa
% if i and j are connected. The degree of connectity if equal to "alfa"

% NOTICE: L=[3*nnode, 3*nnode]

%---
nface=size(element,1);

% loop over all elements
id=[1 2
    1 3
    3 2];

% set initial values for sparse matrices
X=zeros(1,nface*6*3);
irow=zeros(1,nface*6*3);
jcol=zeros(1,nface*6*3);

count=1;
for i=1:nface
    face=element(i,:); 
    
    for j=1:3
        jj=id(j,1);
        zz=id(j,2);
        
            % A(face(jj),face(zz))=1; 

            %------------
            % x-component
            X(count)=1;
            irow(count)=face(jj);
            jcol(count)=face(zz);
            count=count+1;
            
            % symmetric condition ( edge (i,j) == edge (j,i) )
            X(count)=1;
            irow(count)=face(zz);
            jcol(count)=face(jj);
            count=count+1;
            
            %------------
            % y-component
            X(count)=1;
            irow(count)=face(jj)+nnode;
            jcol(count)=face(zz)+nnode;
            count=count+1;
            
            % symmetric condition ( edge (i,j) == edge (j,i) )
            X(count)=1;
            irow(count)=face(zz)+nnode;
            jcol(count)=face(jj)+nnode;
            count=count+1;
            
            %------------
            % z-component
            X(count)=1;
            irow(count)=face(jj)+nnode*2;
            jcol(count)=face(zz)+nnode*2;
            count=count+1;
            
            % symmetric condition ( edge (i,j) == edge (j,i) )
            X(count)=1;
            irow(count)=face(zz)+nnode*2;
            jcol(count)=face(jj)+nnode*2;
            count=count+1;
                        
    end
    
end

% fill sparse matrix
A=femSparse(irow,...
             jcol, ...
             X, ...
             3*nnode, 3*nnode);




