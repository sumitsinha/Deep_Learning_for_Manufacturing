function [fint, vint]=plane2MeshIntersect(Pp, Np, f, v)

% Pp: point on plane
% Np: vector (normalised) normal to plane
% f: face ids
% v: xyz mesh coordinates

fint=[];
vint=v;

% loop over all trias
nf=size(f,1);
for i=1:nf
    
    fi=f(i,:);
    [vertexi, vids, flagi]=plane2TriangleIntersect(Pp, Np, v, fi);

    if flagi==1
      fint=[fint
            fi];
    elseif flagi==2
        
      nv=size(vint,1);
      fint=[fint
            vids(1) nv+1 vids(2)
            vids(2) nv+2 nv+1];
    end
    
    vint=[vint
          vertexi];
end


%-----------------------
%-----------------------
function [vertex, vids, flag]=plane2TriangleIntersect(Pp, Np, P, face)

% flag=1/2/3 => below plane/intersection/above plane

eps=1e-8;

edge=[face(1) face(2)
      face(2) face(3)
      face(3) face(1)];
       
% check intersections
d=zeros(1,3);
Pi=P(face,:);
for i=1:3
    d(i)=distPlane(Pp, Np, Pi(i,:));
end

if d(1)>=eps && d(2)>=eps && d(3)>=eps % above plane
    flag=3;
    vertex=[];
    vids=[];
elseif d(1)<=eps && d(2)<=eps && d(3)<=eps % below plane
    flag=1;
    vertex=[];
    vids=[];
else
    flag=2;
    [vertex, vids]=getIntersection(Pp, Np, P, edge);
end

%--
function [vertex, vids]=getIntersection(Pp, Np, P, edge)

vertex=[];
vids=[];
 
for i=1:3
   
   s=edge(i,1); 
   e=edge(i,2); 
   
   %
   Ps=P(s,:);
   ds=distPlane(Pp, Np, Ps); 
   
   %
   Pe=P(e,:);
   de=distPlane(Pp, Np, Pe); 
   
   if (ds>=eps && de<=eps) || (ds<=eps && de>=eps)
       
        Nl=(Pe-Ps)/norm(Pe-Ps);
        Pint=plane2Line(Pp, Np, Nl, Ps);
        
        vertex=[vertex
                Pint];
            
        if ds<=eps
            vids=[vids, s];
        end
        
        if de<=eps                
            vids=[vids, e]; 
        end
   end
end


%--
function d=distPlane(Pp, Np, Pi)

d=dot( (Pi-Pp), Np );


%--
function Pint=plane2Line(Pp, Np, Nl, Pl)

t=dot( (Pp-Pl), Np )/dot( Nl, Np );

Pint=Pl+t*Nl;




