% crea gli indici dei dof del nodo "idnode"
function index=getIndexNode(idnode,ndof)
  % idnode: node id
  % ndof: # of dofs per node
  % index: dofs list
  
  nn=length(idnode);
  index=zeros(nn,ndof);
  
  start=(idnode-1)*ndof; 
  
  for i=1:ndof
    index(:,i)=start+i; 
  end
  