% solve 3D laplace equation on Box domain 
function B=laplace3D(B, dx, dy, dz)

% INPUT
% B: 3D data matrix
% dx/dy/dz: grid size in row, column and height

% OUTPUT
% B: update data matrix

%-------------------------------------------------------------
% NB:
    % 1. any point in B is assumed as constraint if B(i, j)~=0;
    % 2. use penalty method to enforce constraints
%-------------------------------------------------------------

%-----------------------------------------------------------------------

% define penalty stiffness for solving linear equations
penalty=1e8;

% get boundary conditions
[b, bid]=getBoundaryConstraint(B);

nbc=length(bid); % no. of boundary constraint points

% total no. of grid points
[n, m, p]=size(B);

nt=n*m*p;

% no. of not zero numbers in sparse

% vertex = 6 coeff. (8 nodes) => nv=8;
nv=8;

% boundary edges = 7 coeff. ( 12 edges; nodes per edge 4*(n/m/p-2) ) ; nbe=4*(n+m+p-6)
nbe=4*(n+m+p-6);

% boundary faces = 8 coeff. nbf=2*(m*n-2*(n+m-2)) + 2*(p*n-2*(n+p-2)) + 2*(m*p-2*(p+m-2));
nbf=2*(m*n-2*(n+m-2)) + 2*(p*n-2*(n+p-2)) + 2*(m*p-2*(p+m-2));

% internal points = 9 coeff. nip=nt-nbf-nbe-nv;
nip=nt-nbf-nbe-nv;

nznum=nip*9+nbf*8+nbe*7+nv*6 + nbc; % plus no. of constraints

% allocate space for sparse matrix construction
irow=zeros(nznum,1);
jcol=zeros(nznum,1);
X=zeros(nznum,1);

% get squared grid size
dx2=dx^2;
dy2=dy^2;
dz2=dz^2;

count=1;
counteq=1;
for k=1:p
    for j=1:m
        for i=1:n
                
           % work on X component 
           [irow, jcol, X, count]=getXComponent(i, j, k, n, m,...
                                                 irow, jcol, X, count, counteq, dx2);
                                             
           % work on Y component
           [irow, jcol, X, count]=getYComponent(i, j, k, n, m,...
                                                 irow, jcol, X, count, counteq, dy2);
                                             
            % work on Z component
            [irow, jcol, X, count]=getZComponent(i, j, k, n, m, p,...
                                                 irow, jcol, X, count, counteq, dz2);
                                             
          
            % update counter of equations
            counteq=counteq+1;
            
        end % end i
           
    end % end j
end % end k


% loop over boundary constraint points

% initialize vector of force loads
Fe=zeros(nt,1);
for i=1:nbc
    irow(count)=bid(i);
    jcol(count)=bid(i);
    
    X(count)=penalty;
    
    % vector of force loads
    Fe(bid(i))=b(i)*penalty;
    
    count=count+1;
end

% build assembly matrix
A=sparse(irow, jcol, X, nt, nt);

% clear local variables
clear X
clear irow
clear jcol

% solve linear system

% STEP 1: check how much memory is requred to store the coefficient matrix
mreq=whos('A');

% STEP 1: check the maximum allowed memory for single array/matrix
maxmem = memory;

if mreq.bytes*mreq.bytes <= maxmem.MaxPossibleArrayBytes % use direct solver

    disp('...using Direct solver')
    
    % x=umfpack2(A,'\',Fe);
    x=A\Fe;
    
else % use iterative solver

    disp('...using Iterative solver')
    
    x= gmres(A,Fe,150,1e-13,150);
    
end

% restore matrix
B=reshape(x,n,m,p);


% work on X component
function [irow, jcol, X, count]=getXComponent(i, j, k, n, m,...
                                                 irow, jcol, X, count, counteq, delta)


if i==1

    % get linear index
    t=getLinearIndex3D(i+1, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;
elseif i==n

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i-1, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;
else

    % get linear index
    t=getLinearIndex3D(i+1, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1; 

    % get linear index
    t=getLinearIndex3D(i-1, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1; 
end


% work on Y component
function [irow, jcol, X, count]=getYComponent(i, j, k, n, m,...
                                                 irow, jcol, X, count, counteq, delta)


if j==1

    % get linear index
    t=getLinearIndex3D(i, j+1, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;
elseif j==m

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j-1, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;
else

    % get linear index
    t=getLinearIndex3D(i, j+1, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1; 

    % get linear index
    t=getLinearIndex3D(i, j-1, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1; 
end

% work on Z component
function [irow, jcol, X, count]=getZComponent(i, j, k, n, m, p,...
                                                 irow, jcol, X, count, counteq, delta)


if k==1

    % get linear index
    t=getLinearIndex3D(i, j, k+1, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;
elseif k==p

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k-1, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;
else

    % get linear index
    t=getLinearIndex3D(i, j, k+1, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1;

    % get linear index
    t=getLinearIndex3D(i, j, k, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=-2/delta;
    count=count+1; 

    % get linear index
    t=getLinearIndex3D(i, j, k-1, n, m);

    irow(count)=counteq;
    jcol(count)=t;
    X(count)=1/delta;
    count=count+1; 
end

% get constraint value and ids
function [b, bid]=getBoundaryConstraint(B)

% INPUT
% B: data matrix

% OUTPUT
% b: list of b. constraints
% bid: related ids

%-------------------------------------------------------------
% NB: any point in B is assumed as constraint if B(i, j, k)~=0;
%-------------------------------------------------------------

b=[];
bid=[];

[n, m, p]=size(B);

count=1;
for i=1:n
    for j=1:m
        for k=1:p
    
            if B(i, j, k)~=0

               % get linear index
               t=getLinearIndex3D(i, j, k, n, m);
               
               b(count)=B(i,j, k);
               bid(count)=t;

               count=count+1;
               
            end
        end
    end
end


