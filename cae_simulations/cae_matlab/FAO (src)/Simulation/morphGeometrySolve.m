% solve morphing mesh
function [D, flag]=morphGeometrySolve(data, idpart)

% INPUT
% data: data structure
% idpart: part ID

% OUTPUT
% D: deviation matrix (nx3)
% flag=0/1 (part not solved; solved)

% NOTICE: dp(Q)=Y*M. Y weight matrix. M: projection matrix
%   M(:,j)=inv(X)*Dj, j=x,y,z... 

eps=1e-6;

flag=1;
D=[];

%------------------------------
if data.Input.Part(idpart).Status~=0 && ~data.Input.Part(idpart).Enable
    flag=0;
    return
end

%--
seardist=data.Model.Variation.Option.SearchDist;

% node: nodes to be morphed (nx3)
% Pc: control points (rx3)
% dPc: deviation of control points (rx1)
% Nc: direction of morphing (rx3)

% read nodes
idnode=data.Model.Nominal.Domain(idpart).Node;
n=length(idnode); % no. of nodes

node=data.Model.Nominal.xMesh.Node.Coordinate(idnode,:);

% no. of control points
r=length(data.Input.Part(idpart).Morphing);

% read inputs
Pc=zeros(r,3);
Nc=zeros(r,3);
dPc=zeros(r,1);
for i=1:r
        
    % control point
    Pc(i,:)=data.Input.Part(idpart).Morphing(i).Pc;
    
    % normal
    mnormal=data.Input.Part(idpart).Morphing(i).NormalMode{1};
    
    if mnormal==2 % use model
        [~, Nci, flagi]=point2PointNormalProjection(data.Model.Nominal, Pc(i,:), idpart, seardist);
               
       if ~flagi % use user settings
           Nci=data.Input.Part(idpart).Morphing(i).Nc; 
           warning('Warning (morphing) - failed to calculate normal vector @ control point [%g]', i)
       end
       
    else % user
       Nci=data.Input.Part(idpart).Morphing(i).Nc; 
    end
    
    ln=norm(Nci);
    
    if ln<=eps
        warning('Warning (morphing) - failed to calculate normal vector @ control point [%g]', i)
        flag=0;
        return
    end
    
    Nci=Nci/ln;
    Nc(i,:)=Nci; %--------------
                
    % deviation at control point
    dPc(i,:)=data.Input.Part(idpart).Morphing(i).DeltaPc;
        
end

% initialise X and Y
X=zeros(r,r);
Y=zeros(n,r);
deltaPc=zeros(r,3);

% STEP 1: calculate X and Y
for i=1:r   
    
    % get selection domain
    [Pi, radiusi, Roti]=localgetEllipsoid(data, idpart, node, i);

    %--
    X(:,i)=evalQFunction(Pc, radiusi, Pc(i,:), Pi, Roti); 
    
    %--
    Y(:,i)=evalQFunction(node, radiusi, Pc(i,:), Pi, Roti); 
    
    %--
    deltaPc(i,:)=Nc(i,:)*dPc(i);        
end
         
% STEP 2: get projection matrix
M=X\deltaPc; % solve the least square problem

% STEP 3: get deviations
D=Y*M;


%------------

% get weight
function Q=evalQFunction(Point, radius, Pc, C, RotC)

% Point: point to be evauated
% radius: radius of the ellipsoid
% Pc: control point
% C: centre of the ellipsoid (influence hull)
% RotC: rotation matrix of the ellipsoid (influence hull)

distVal=evalDistPt_ellipsoid(Point, radius, Pc, C, RotC);

Q=evalpolyFnc(distVal); 

% get distance
function Q=evalpolyFnc(distVal)

% calculate 3rd order polinomial passing by P1 and P2 and with orizzontal slope at P1 e P2...
P1=[0 1];
P2=[1 0];
T1=0; % orizzontal slop at P1
T2=0; % orizzontal slop at P2

A=[1 P1(1) P1(1)^2 P1(1)^3
   1 P2(1) P2(1)^2 P2(1)^3
   0 1     2*P1(1) 3*P1(1)^2
   0 1     2*P2(1) 3*P2(1)^2];

B=[P1(2)
   P2(2)
   T1
   T2];

P=A\B; % solve least square problem

% get weight
Q=P(1)+P(2)*distVal+P(3)*distVal.^2+P(4)*distVal.^3;


%---------------
function [Pi, radiusi, Roti]=localgetEllipsoid(data, idpart, node, ir)

idsele=data.Input.Part(idpart).Morphing(ir).Selection;

if idsele==0
    dV=getBoundingVolume(node);
else
   [dV, ~]=retrieveStructure(data, 'Selection', idsele);
end

% extract features
Pi=dV.Pm;
radiusi=dV.Rm;

Nc1=dV.Nm1;
Nc2=dV.Nm2;

Z=cross(Nc1, Nc2);
Z=Z/norm(Z);
Y=cross(Z, Nc1);
Roti = [Nc1', Y', Z']; 
