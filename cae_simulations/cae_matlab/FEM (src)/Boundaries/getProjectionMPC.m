% compute parameters for MPC
function [flag,...
          dofs,...
          coefu,...
          coefv,...
          coefw,...
          gsign,...
          Pp]=getProjectionMPC(fem, P0, N0, idpart, dsearch, frame)
 
% flag: true/false => proejction found or not
% dofs: dofs involved in the constraint: [u, v, w, rotx, roty, rotz]
% coef: coefficient to be included into MPC formulation (x, y, z)
% gsign: signed gap
% Pp: projected master node
 
% P0/N0: point and vector
% idpart: id part for projection
% dsearch: searching distance
% frame: used for calculating the projected point

% init
dofs=[];
coefu=[];
coefv=[];
coefw=[];
[Pp, flag, gsign, edata]=pointNormal2PointProjection(fem, P0, N0, idpart); % MEX file
% checks
if ~flag
    return
end
if abs(gsign)>dsearch
    flag=false;
    return
end
%
% get dofs ids
dofs=fem.xMesh.Element(edata(1)).ElementNodeIndex; 

% get coefficients
idm=fem.xMesh.Element(edata(1)).Element;
if strcmp(frame,'ref')
    % get node coordinates of the element
    Pmb0=fem.xMesh.Node.Coordinate(idm,:);
    % get Rotation matrix
    R0l=fem.xMesh.Element(edata(1)).Tmatrix.T0lGeom;
    % get origin of local coordinate frame
    Pe=fem.xMesh.Element(edata(1)).Tmatrix.P0lGeom;
elseif strcmp(frame,'def')
    % get node coordinates of the element
    Pmb0=fem.Sol.DeformedFrame.Node.Coordinate(idm,:);
    % get Rotation matrix
    R0l=fem.Sol.DeformedFrame.Element(edata(1)).Tmatrix.T0lGeom;
    % get origin of local coordinate frame
    Pe=fem.Sol.DeformedFrame.Element(edata(1)).Tmatrix.P0lGeom;
end
% transform nodes
Pmb=applyinv4x4(Pmb0, R0l, Pe);
Pxy=Pp;  
Pxy=applyinv4x4(Pxy, R0l, Pe);
% get coefficient
if edata(2)==1 % quad
 [csips,etaps]=mapxy2csietaQuad4(Pxy(1),Pxy(2),Pmb);

 % ... and related weight
 [N, ~]=getNdNquad4node(csips,etaps);
elseif edata(2)==2 % tria
 [csips,etaps]=mapxy2csietaTria3(Pxy(1),Pxy(2),Pmb);

 % ... and related weight
 [N, ~]=getNdNtria3node(csips,etaps);

 %.........................

end 
coefu=N0(1)*N';
coefv=N0(2)*N';
coefw=N0(3)*N';
