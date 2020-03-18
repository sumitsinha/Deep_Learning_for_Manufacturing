%...             
function [X,Y,Z]=renderCylObj(rc, ll, lu, Nc, Pc, res)

%--
if nargin==5
    res=10;
end

[X, Y, Z]=createCylinder(rc, res);

%...
Z(1,:)=ll;
Z(2,:)=lu;

% get base:
NS=null(Nc);

Nx=NS(:,1);
Ny=cross(Nc,Nx);

R=[Nx,Ny',Nc'];

%...
P1=[X(1,:);Y(1,:);Z(1,:)]';
P2=[X(2,:);Y(2,:);Z(2,:)]';

P1=apply4x4(P1, R, Pc)';

P2=apply4x4(P2, R, Pc)';

%... then
X=[P1(1,:);P2(1,:)];
Y=[P1(2,:);P2(2,:)];
Z=[P1(3,:);P2(3,:)];

% creat cylinder using parametric cylinder equation
function [X, Y, Z]=createCylinder(rc, res)

theta=linspace(0, 2*pi, res);
z=[0 1];

[THETA, Z]=meshgrid(theta, z);

X=rc.*cos(THETA);
Y=rc.*sin(THETA);

