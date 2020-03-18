function L=element2Laplacian(element, node)

% INPUT
% element: element connection
% node: xyz coordinates of mesh nodes

% OUTPUT
% L(i,j) = sum( cot(alphaij) ) where alphaij is the adjacent angle to edge (i,j) for all edges belonging to the node-ring

%---
% get no. of nodes
nnode=size(node,1);

% get ring connections
ring=node2ElementRing(element, nnode);

% loop over all node
L=sparse(nnode,nnode);

for i=1:nnode
    
    % get connected elements
    for r=ring{i}
        
        % current connected element
        facer=element(r,:);
        
        if facer(1)==i
            v = facer(2:3);
        elseif facer(2)==i
            v = facer([1 3]);
        elseif facer(3)==i
            v = facer(1:2);
        end
        
        j = v(1); k = v(2);
        
        % get coordinates
        vi = node(i,:);
        vj = node(j,:);
        vk = node(k,:);
        
        % get angles
        alphaij = getangle(vk-vi,vk-vj);
        betaij = getangle(vj-vi,vj-vk);
        
        % update laplacian (add single contributions)
        L(i,j)=L(i,j) + cot(alphaij);
        L(i,k)=L(i,k) + cot(betaij);
        
    end
    
end

function angle=getangle(V1, V2)

l1=norm(V1);
l2=norm(V2);

angle=acos( dot(V1, V2) ) / (l1*l2);


