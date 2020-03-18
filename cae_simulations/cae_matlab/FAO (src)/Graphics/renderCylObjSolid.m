%...             
function [f, v]=renderCylObjSolid(radius, L, Nc, Tc, Vc, Pc, res, angle, sign)

L=L*sign;

% calculate rotation matrix
Rc=[Tc', Vc', Nc'];

% external rotation around Z axis 
angle=angle*pi/180;

% build rotation around Z
R0z=[cos(angle) -sin(angle) 0
     sin(angle) cos(angle)  0
     0       0        1];

Rc=Rc*R0z;

% define coordinate of base
t=linspace(0,2*pi,res);

%---
x=radius*cos(t);
x(end)=[];
y=radius*sin(t);
y(end)=[];
z=ones(length(t),1);
z(end)=[];

vl=[x' y' z*0.0];
vu=[x' y' z*L];
v=[vl
   vu
   mean(vl,1)
   mean(vu,1)];

% wrap face
f=localWrapFace(res);

if sign<0
    f(:, [3 2])=f(:, [2 3]);
end


%... transform back into global frame
v=apply4x4(v, Rc, Pc);


%---------
function f=localWrapFace(res)

% wall
f=[];
for i=1:res-1
    j=i+1;
    if j>res-1
        j=1;
    end
    f=[f
       i j (res-1)+j
       i (res-1)+j (res-1)+i ];
end

% caps
for i=1:res-1
    j=i+1;
    if j>res-1
        j=1;
    end
    f=[f
       (2*res-2)+1 j i
       (2*res-2)+2 (res-1)+i (res-1)+j];
end


