%...             
function [X,Y,Z]=renderEllipsoidObj(rc, Pc, Rc)

% rc: radius
% Pc: center
% Rc: rotation matrix

res=30;
[X,Y,Z]=ellipsoid(0, 0, 0,...
        rc(1), rc(2), rc(3),...
        res);

%...
P=[X(:),Y(:),Z(:)];
P=apply4x4(P, Rc, Pc);

%... then
X=reshape(P(:,1),res+1,res+1);
Y=reshape(P(:,2),res+1,res+1);
Z=reshape(P(:,3),res+1,res+1);