% get ids of boundary nodes
function bnode=getBoundaryNode(fem)

% INPUT
% fem:fem structure

% OUTPUT
% bnodes: list of boundary nodes ids

% N.B: it finds out the nodes between selected and un-selected regions

% get adjacency matrix
fprintf('    FEMP Boundary - calculating connections...\n')
A=element2Adjacent(fem);

fprintf('    FEMP Boundary - calculating boundaries...\n')

% loop over all nodes
[bnode, ~]=find(A==1);

% remove duplicates
bnode=unique(bnode);





