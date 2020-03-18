%...             
function [X,Y,Z]=createPinHoleObj(radius,N1,N2, d, tx, ty, alfa,beta)

res=10;
[X,Y,Z]=cylinder(radius,res);

%...
Z(1,:)=-radius;
Z(2,:)=radius;

X=X+tx;
Y=Y+ty;

% get base:
N3=cross(N1,N2);
N3=N3/norm(N3);

R=[N1',N2',N3'];

Rx=RodriguesRot([1 0 0],alfa);
Ry=RodriguesRot([0 1 0],beta);

R=R*Rx*Ry;

%...
P1=[X(1,:);Y(1,:);Z(1,:)]';
P2=[X(2,:);Y(2,:);Z(2,:)]';

P1=apply4x4(P1, R, d)';

P2=apply4x4(P2, R, d)';

%... then
X=[P1(1,:);P2(1,:)];
Y=[P1(2,:);P2(2,:)];
Z=[P1(3,:);P2(3,:)];