%---------------------------
% renumber mesh structure
function elements=renumberElements(elements, idnode)

[m, n]=size(elements);
nnode=length(idnode);

% loop over element ids
for i=1:m*n
    nodei=elements(i);
    
      % loop over all nodes
      for j=1:nnode
         if nodei == idnode(j)
             elements(i)=j; 
             break;
         end
      end      
end
