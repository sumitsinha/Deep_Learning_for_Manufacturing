%...             
function [f, v]=renderLShapeObjSolid(A, B, C, L, Nc, Tc, Vc, Pc, angle, sign)

% angle: angle in [deg]

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
vl=[-A/2 -B/2 0
    A/2 -B/2 0
    A/2 -B/2 2/3*L
    A/2+C -B/2 2/3*L
    A/2+C -B/2 L
    -A/2 -B/2 L];
vu=[-A/2 B/2 0
    A/2 B/2 0
    A/2 B/2 2/3*L
    A/2+C B/2 2/3*L
    A/2+C B/2 L
    -A/2 B/2 L];
v=[vl
   vu];

f=localWrapFace();

if sign<0
    f(:, [3 2])=f(:, [2 3]);
end

%... transform back into global frame
v=apply4x4(v, Rc, Pc);

%----------------
function f=localWrapFace

% wall 1
f=[1 2 3
   1 3 6
   5 6 3
   4 5 3];

% wall 2
f=[f
   7 9 8
   7 12 9
   9 12 11
   9 11 10];

% wall 3
f=[f
   7 1 6
   7 6 12];

% wall 4
f=[f
   2 8 3
   8 9 3];

% wall 5
f=[f
   4 10 5
   5 10 11];

% wall 6
f=[f
   1 8 2
   1 7 8];

% wall 7
f=[f
   6 5 11
   6 11 12];

% wall 8
f=[f
   3 10 4
   3 9 10];
           