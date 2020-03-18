%... 
function [f, v]=renderPrismaObjSolid(rc, Pc, N1, N2)

% calculate rotation matrix
Z=cross(N1, N2);
Z=Z/norm(Z);
Y=cross(Z, N1);

Rc = [N1', Y', Z'];  

%--
A=rc(1);
B=rc(2);
L=rc(3);

% define coordinate of base
vl=[A/2 B/2 -L/2
    -A/2 B/2 -L/2
    -A/2 -B/2 -L/2
    A/2 -B/2 -L/2];
vu=[A/2 B/2 L/2
    -A/2 B/2 L/2
    -A/2 -B/2 L/2
    A/2 -B/2 L/2];
v=[vl
   vu];

% wall
f=[];
for i=1:4
    j=i+1;
    if j>4
        j=1;
    end
    f=[f
       i j 4+j
       i 4+j 4+i];
end

% caps
f=[f
   1 2 3
   1 3 4
   5 6 7
   5 7 8];

%... transform back into global frame
v=apply4x4(v, Rc, Pc);
