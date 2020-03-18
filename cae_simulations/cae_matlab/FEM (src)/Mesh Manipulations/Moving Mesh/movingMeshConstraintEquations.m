function [C, q]=movingMeshConstraintEquations(Panchor, Nanchor, idanchor, dev, nnode)

% INPUT:
% L0: laplacian matrix
% Panchor: xyz of anchor points
% Nanchor: constraint direction of the anchor points
% idanchor: ids of anchor points
% dev: list of deviations
% nnode: no. of total nodes

% OUTPUT:
% L: assembled laplacian matrix
% C: matrix of constraint coefficient
% q: list of prescribed constraints

% write C and q
nc=size(Panchor,1);

q=zeros(3*nc,1);

% variables for sparse matrices
X=zeros(1,3*nc);
irow=zeros(1,3*nc);
jcol=zeros(1,3*nc);

% loop over all constraints
count=1;
for i=1:nc
    
  di=dev(i);
  Ni=Nanchor(i,:);
  
  vi=Panchor(i,:);
  
  % new point on the line passing thought vi along Ni
  pi=vi+di*Ni;
  
  % write prescribed displacement
  q(count)=pi(1); % x
  q(count+1)=pi(2); % y
  q(count+2)=pi(3); % z

  % get id constraint
  id=idanchor(i);
  
  % write coefficients (unit weight)
  X(count)=1; % x
  X(count+1)=1; % y
  X(count+2)=1; % z
  
  irow(count)=count;
  irow(count+1)=count+1;
  irow(count+2)=count+2;
  
  jcol(count)=id;
  jcol(count+1)=nnode+id;
  jcol(count+2)=2*nnode+id;
    
  % update counter
  count=count+3;
  
end

% fill sparse matrix
C=femSparse(irow,...
             jcol, ...
             X, ...
             3*nc, nnode*3);
         
