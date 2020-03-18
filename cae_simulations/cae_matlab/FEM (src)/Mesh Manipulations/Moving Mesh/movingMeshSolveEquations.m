function node=movingMeshSolveEquations(L, C, q, node)

% INPUT:
% L: assembled laplacian matrix
% C: matrix of constraint coefficient
% q: list of prescribed constraints
% node: xyz coordinates of mesh nodes

% OUTPUT:
% node: xyz coordinates of mesh nodes

nnode=size(node,1);

% create vector un unknowns
v=[node(:,1)
    node(:,2)
    node(:,3)];

% get delta
delta=L*v;

% built matrix of coefficients
A=(L'*L+C'*C);

% built known vectors
b=(L'*delta+C'*q);

% solve using solver for sparse matrices
x=cholmod2(A,b);
%x=umfpack2(A,'\',b);

% store back (as x | y | z)
node=[x(1:nnode) x(nnode+1:2*nnode) x(2*nnode+1:end)];
