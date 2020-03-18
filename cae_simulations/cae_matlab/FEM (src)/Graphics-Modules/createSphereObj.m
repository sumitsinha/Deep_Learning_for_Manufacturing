%...             
function [X,Y,Z]=createSphereObj(radius,d,res)

if nargin==2
    res=30;
end

[X,Y,Z]=sphere(res);

P=[X(:),Y(:),Z(:)]*radius;

P=apply4x4(P, eye(3,3), d);

%... then
X=reshape(P(:,1),res+1,res+1);
Y=reshape(P(:,2),res+1,res+1);
Z=reshape(P(:,3),res+1,res+1);