% compute mesh weight
function W=movingMeshComputeMeshWeight(element, nnode)

% INPUT
% element: element connection
% nnode: no. of nodes

% OUTPUT
% L(i,j)=1 if vertices i and j are connected; 0 otherwhise
    
% NOTICE: W=[3*nnode, 3*nnode]
    
     
W=movingMeshElement2Adjacent(element, nnode);

W(W>0)=1;
        
% normalize weight matrix
W = speye(nnode*3) - diag(sum(W,2).^(-1)) * W;

        
        