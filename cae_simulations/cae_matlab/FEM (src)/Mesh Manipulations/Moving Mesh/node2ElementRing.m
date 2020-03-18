function ring=node2ElementRing(element, nnode)

% INPUT
% element: element connection
% nnode: no. of mesh nodes

% OUTPUT
% ring{i}=list of all elements connected to vertex "i-th"

nfaces = size(element,1);

ring = cell(1,nnode);

% loop over all elements
for i=1:nfaces
    for k=1:3
        ring{element(i,k)}(end+1) = i;
    end
end