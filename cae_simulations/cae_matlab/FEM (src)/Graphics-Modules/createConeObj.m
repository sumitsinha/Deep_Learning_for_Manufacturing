function [X, Y, Z]=createConeObj(fmin, fmax, alfa, P0, N0)

% alfa: angle in degree

res=15;

%--------
alfa=alfa*pi/180;

rmin=fmin*tan(alfa/2);
rmax=fmax*tan(alfa/2);

r=linspace(rmin,rmax,res);
theta = linspace(0,2*pi,res);
[r,theta] = meshgrid(r,theta);

X = r.*cos(theta);
Y = r.*sin(theta);
Z = repmat(linspace(fmin,fmax,res),res,1);

% get base:
NS=null(N0);

Nx=NS(:,1);
Ny=cross(N0,Nx);

R=[Nx,Ny',N0'];

%...
P=[X(:),Y(:),Z(:)];

P=apply4x4(P, R, P0);

%... then
X=reshape(P(:,1),res,res);
Y=reshape(P(:,2),res,res);
Z=reshape(P(:,3),res,res);
