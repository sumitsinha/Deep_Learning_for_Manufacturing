% Compute auto-scale value for plotting deformed geometry
function sc=computeAutoScaleFactorContour(fem, idpart)

%--
% Scale factor "sc" is computed as ratio between the size of the part and
% the max displacement with a factor of 10

% ARGUMENTs:
%
% Inputs:
% fem: fem model
% idpart: list of parts to be used to extract the scaling factor
%
% Outputs:
% sc: scaling factor
%
if nargin==1
    idpart=1;
end

% Read nodes
idnode=[fem.Domain(idpart).Node];
xyz=fem.xMesh.Node.Coordinate(idnode,:);

% Estimate max size of the part (max distance from the centre)
mxyz=mean(xyz,1);

d(:,1)=xyz(:,1)-mxyz(1);
d(:,2)=xyz(:,2)-mxyz(2);
d(:,3)=xyz(:,3)-mxyz(3);

di=sqrt(sum(d.^2, 2));
xyzmax=max(di);

% Extract "U"
if strcmp(fem.Options.Physics, 'structure')
    dofsnodes=fem.xMesh.Node.NodeIndex(idnode,1:3);
    U(:,1)=fem.Sol.U(dofsnodes(:,1));
    U(:,2)=fem.Sol.U(dofsnodes(:,2));
    U(:,3)=fem.Sol.U(dofsnodes(:,3));

    Ui=sqrt(sum(U.^2, 2));
    Umax=abs(max(Ui));

    % Compute scale factor
    if Umax>0
        sc=(xyzmax/(Umax*10));
    else
        sc=0;
    end
else
    sc=0;
end
