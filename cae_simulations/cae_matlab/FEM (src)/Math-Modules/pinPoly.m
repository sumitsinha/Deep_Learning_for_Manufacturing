% check if point "Pt" is inside polygon "Pp" (works for convex polygon) 
function flag=pinPoly(Pt,Pp,eps)

%...

flag=false;

% x-y plane
Pp=Pp(:,1:2);
Pt=Pt(1:2);
  
% calculate polygon area 
areap=getAreaPoly(Pp);

% add first point at the end
Pp(end+1,:)=Pp(1,:);

np=size(Pp,1);
areac=0;

% get sum of area triangles
for i=1:np-1
    Pi=[Pt
        Pp(i,:)
        Pp(i+1,:)];
    areac=areac+getAreaPoly(Pi);
end

% check
if abs(areac-areap)<=eps
    flag=true;
end
    


