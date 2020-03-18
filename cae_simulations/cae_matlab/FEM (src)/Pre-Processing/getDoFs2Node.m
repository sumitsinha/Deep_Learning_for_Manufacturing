% This function extract DoFs of a given node
function DoFs=getDoFs2Node(idnode,...
                           etype)

% DoFs: list of DoFs
% idele: list of elements connected by idnode

% set initial value
DoFs=[];

% TRIA and QUAD elements (6 dofs per node)
if strcmp(etype, 'shell')

   DoFs=getIndexNode(idnode,6);
           
end

