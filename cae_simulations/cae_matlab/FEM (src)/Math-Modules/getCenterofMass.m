% get baricenter of part "idpart"
function Pg=getCenterofMass(fem, idpart)

% INPUT:
% fem: fem strucuture
% idpart: part identification

% OUTPUT:
% Pg: baricenter = 1/A*sum(Ai*Pgi)
    % A= total surface area
    % Ai and Pgi are the area and the baricenter ofthe element "i-th", respectively

Pg=[0 0 0];

% get elements
ele=fem.Denoise.Domain(idpart).Tria;
nele=length(ele);

% get node coordinates
node=fem.xMesh.Node.Coordinate;

At=0;
for i=1:nele
    
    % get node ids
    idnode=fem.Denoise.Tria(fem.Denoise.Domain(idpart).Tria(i),:);
    
    nodei=node(idnode,:);
    
    % baricenter of the "i-th" tria
    Pgi=mean(nodei,1);
    
    % area of the "i-th" tria
    Ai=getAreaTria3D(nodei);
    
    % update
    At=At+Ai;
    
    % update
    Pg(1)=Pg(1)+Pgi(1)*Ai;
    Pg(2)=Pg(2)+Pgi(2)*Ai;
    Pg(3)=Pg(3)+Pgi(3)*Ai;
    
end

%... so...
Pg=Pg/At;

