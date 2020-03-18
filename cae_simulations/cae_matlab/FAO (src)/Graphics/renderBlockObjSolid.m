%... 
function [f, v]=renderBlockObjSolid(A, B, L, Nc, Tc, Vc, Pc, angle, sign)

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
vl=[A/2 B/2 0
    -A/2 B/2 0
    -A/2 -B/2 0
    A/2 -B/2 0];
vu=[A/2 B/2 L
    -A/2 B/2 L
    -A/2 -B/2 L
    A/2 -B/2 L];
v=[vl
   vu];

% wrap face
f=localWrapFace();

if sign<0
    f(:, [3 2])=f(:, [2 3]);
end

%... transform back into global frame
v=apply4x4(v, Rc, Pc);

           
%---------
function f=localWrapFace()
        
f=[1 2 6
   1 6 5
   4 1 5
   4 5 8
   3 4 8
   3 8 7
   2 3 7
   2 7 6
   5 6 7
   5 7 8
   4 3 1
   3 2 1];

