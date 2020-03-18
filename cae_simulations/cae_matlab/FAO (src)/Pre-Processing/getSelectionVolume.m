%--
function idnode=getSelectionVolume(data, idpart, idselection)

% data: model
% idpart: part to be used for selection
% idselection: ID of current selection in "data.Input.Selection"

% idnode: list of selected nodes

idnode=[];

% get selection
if ~isfield(data.Input, 'Selection')
    return
end

fselection=data.Input.Selection(idselection);

Pc=fselection.Pm;
 
Nc1=fselection.Nm1;
Nc2=fselection.Nm2;
rc=fselection.Rm;
type=fselection.Type{1};
 
%- get points
idnode=data.Model.Nominal.Domain(idpart).Node;
nodes=data.Model.Nominal.xMesh.Node.Coordinate(idnode,:);

%------------------------
% check selection
inblock=inBlock3D(rc, Pc, Nc1, Nc2, nodes, type);

% update structure
idnode=idnode(inblock);


%-----------
function inblock=inBlock3D(rc, Pc, N1, N2, point, type)

% type=1/2 => prisma/ellipsoid

% calculate rotation matrix
Z=cross(N1, N2);
Z=Z/norm(Z);
Y=cross(Z, N1);

Rc = [N1', Y', Z'];  

% init output
n=size(point,1);
inblock=false(1,n);

% back to local frame
point=applyinv4x4(point, Rc, Pc);

% loop over points
for i=1:n
    
    % point i-th
    x=point(i,1); y=point(i,2); z=point(i,3); 
    
    if type==1 % prisma
        if (x>=-rc(1)/2 && x<=rc(1)/2) && (y>=-rc(2)/2 && y<=rc(2)/2) && (z>=-rc(3)/2 && z<=rc(3)/2) % check if point is inside the volume
            inblock(i)=true;
        end
    elseif type==2 % ellipsoid
        if (x^2/rc(1)^2 + y^2/rc(2)^2 + z^2/rc(3)^2)<=1 % check if point is inside the volume
            inblock(i)=true;
        end
    end
end
