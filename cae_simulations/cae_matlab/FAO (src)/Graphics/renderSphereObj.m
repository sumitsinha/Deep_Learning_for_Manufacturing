%...             
function [X,Y,Z]=renderSphereObj(rc, Pc)

res=10;
[X,Y,Z]=sphere(res);

%...
P=[X(:),Y(:),Z(:)]*rc;
P=apply4x4(P, eye(3,3), Pc);

%... then
X=reshape(P(:,1),res+1,res+1);
Y=reshape(P(:,2),res+1,res+1);
Z=reshape(P(:,3),res+1,res+1);