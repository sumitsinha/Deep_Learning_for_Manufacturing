%...             
function [X,Y,Z]=createCylObjGeneral(radius,len, Nz,d)

res=10;
[X,Y,Z]=cylinder(radius,res);

%...
Z(1,:)=0;
Z(2,:)=len;

% get base:
NS=null(Nz);

Nx=NS(:,1);
Ny=cross(Nz,Nx);

R=[Nx,Ny',Nz'];

%...
P1=[X(1,:);Y(1,:);Z(1,:)]';
P2=[X(2,:);Y(2,:);Z(2,:)]';

P1=apply4x4(P1, R, d)';

P2=apply4x4(P2, R, d)';

%... then
X=[P1(1,:);P2(1,:)];
Y=[P1(2,:);P2(2,:)];
Z=[P1(3,:);P2(3,:)];