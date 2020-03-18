%...             
function [X,Y,Z]=renderSlotObj(radius,len, Nx, Ny, d)

% Height of the slot is assumed "2*radius"
res=10;

% cyl 1
teta=-90:res:90;
teta=teta*pi/180;
z=[-radius radius];
[t,Z1]=meshgrid(teta,z);

Y1=radius*cos(t)+len/2;
X1=radius*sin(t);

% cyl 2
teta=90:res:270;
teta=teta*pi/180;
[t,Z2]=meshgrid(teta,z);

Y2=radius*cos(t)-len/2;
X2=radius*sin(t);

% store all
X=[X1,X2, X1];
Y=[Y1,Y2, Y1];
Z=[Z1,Z2, Z1];

% get base:
Nz=cross(Nx,Ny);

R=[Nx',Ny',Nz'];

%...
P1=[X(1,:);Y(1,:);Z(1,:)]';
P2=[X(2,:);Y(2,:);Z(2,:)]';

P1=apply4x4(P1, R, d)';

P2=apply4x4(P2, R, d)';

%... then
X=[P1(1,:);P2(1,:)];
Y=[P1(2,:);P2(2,:)];
Z=[P1(3,:);P2(3,:)];