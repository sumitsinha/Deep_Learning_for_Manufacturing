function [X, Y, Z]=renderSpherePatchObj(radius, teta, phi, P0, N0, res)

% radius: radius
% teta/phi: angle in degree (min/max)
% P0/N0: position and cone axis

teta=teta.*pi/180;
phi=phi.*pi/180;

t=linspace(teta(1),teta(2),res);
p = linspace(phi(1),phi(2),res);
[t,p] = meshgrid(t, p);

X = radius.*cos(t) .* cos(p);
Y = radius.*cos(t) .* sin(p);
Z = radius.*sin(t);

% get base:
R=vector2Rotation(N0);

%...
P=[X(:),Y(:),Z(:)];

P=apply4x4(P, R, P0);

%... then
X=reshape(P(:,1),res,res);
Y=reshape(P(:,2),res,res);
Z=reshape(P(:,3),res,res);

