% calculate natural coordinate from real coordinate
function [csi,eta]=mapxy2csietaTria3(xp,yp,P)

% Partially inspired on:
%...................................................................................
% http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch23.d/IFEM.Ch23.pdf
%...................................................................................

% get coordinates
x=P(:,1);
y=P(:,2);

% costanti
ax=xp-x(1);
bx=x(2)-x(1);
cx=x(3)-x(1);

ay=yp-y(1);
by=y(2)-y(1);
cy=y(3)-y(1);

% get natural coordinates
eta=(ay*bx-ax*by)/(cy*bx-cx*by);
csi=(ax-eta*cx)/bx;