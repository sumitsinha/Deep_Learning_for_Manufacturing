% calculate rotation matrix for tria elements
function R=getR0lTria(P)

% P=node coordinates
% R=rotation matrix

% get tangent vectors located at point (csi=0, eta=0)
x=P(2,:)-P(1,:);
x=x/norm(x);

ty=P(3,:)-P(1,:);

% get normal vector
z=cross(x,ty);
z=z/norm(z);

% get y
y=cross(z,x);

% ... then, rotation matrix becomes
R=[x' y' z'];