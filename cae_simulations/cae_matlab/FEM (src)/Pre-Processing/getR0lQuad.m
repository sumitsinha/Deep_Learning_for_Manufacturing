% calculate rotation matrix for quad elements
function R=getR0lQuad(P)

% Based on "Finite Rotations Shells (by K. Wisniewski - 2010)

% P=node coordinates
% R=rotation matrix

% get tangent vectors located at point (csi=0, eta=0)
t1=[1/4 -1/4 -1/4 1/4]*P;
t1=t1/norm(t1);

g2=[1/4 1/4 -1/4 -1/4]*P;

% get normal vector
t3=cross(t1,g2);
t3=t3/norm(t3);

% get t2
t2=cross(t3,t1);

% ... then, rotation matrix becomes
R=[t1' t2' t3'];