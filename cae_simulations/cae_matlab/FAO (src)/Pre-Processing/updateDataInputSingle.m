% update input structure
function data=updateDataInputSingle(data, field, id, opt)

% INPUT
% data: data structure
% field: field identification name
% id:  field identification ID
% opt: [logical]
    % opt(1): true/false => true: refresh all inputs (recompute projections, recompute parametrisations)
    % opt(2): true/false => true: refresh only part features (recompute projections, recompute parametrisations)
    % opt(3): true/false => true: apply placement
    % opt(4): true/false => true: project Pm on geometry (the projected point will be used as "reference")

% Note: f.Master/Slave => used for projection
% Note: f.DomainM/DomainS => used for calculation by FEM kernel

% OUTPUT
% data: updated data structure

%------------------------------
% ex: -1/0/1/2/3 = not updated/success/failed to project on master/failed to project on slave/wrong input

if nargin==3
    opt=true(1,4);
elseif nargin==4
    if length(opt)<4
        opt(4)=0;
                        % error('Model build (input): wrong input format');
    end
end

% get field
[f, ~]=retrieveStructure(data, field, id);

if strcmp(field, 'Robot')
    f=initRobot(f);
    f.Status{1}=0;
    data=retrieveBackStructure(data, f, field, id);
elseif strcmp(field, 'Contact')
    % Init
    f.Status=[];
    f.Status{1}=0;
    data=checkMasterSlave(data, f, field, id);
    return
else
    if opt(2)
        if strcmp(field, 'Contact')
            % Init
            f.Status=[];
            f.Status{1}=0;
            data=checkMasterSlave(data, f, field, id);
            return
        elseif strcmp(field, 'Stitch') || strcmp(field, 'Hole') || strcmp(field, 'Slot')
            data=runUpdate(data, f, field, id, opt);
        end
    elseif opt(1)
        data=runUpdate(data, f, field, id, opt);
    end
end

%--
function data=runUpdate(data, f, field, id, opt)

% Init/reset
f.Status=[];
f.Status{1}=0;

if ~f.Enable
    ex=1;

    for i=1:length(f.Status)
        f.Status{i}=ex;
    end

    data=retrieveBackStructure(data, f, field, id);
    return
end
    
if ~isfield(data.Input,'Part')
    ex=3;

    for i=1:length(f.Status)
        f.Status{i}=ex;
    end

    data=retrieveBackStructure(data, f, field, id);
    return
end
%--    
% check master part
[f, flagpass]=checkMasterStatus(data, f);
if ~flagpass
    data=retrieveBackStructure(data, f, field, id);
    return
end

% check slave part (if any)
if strcmp(field,'Stitch') || strcmp(field,'ClampM')

    [f, flagpass]=checkSlaveStatus(data, f);
    if ~flagpass
        data=retrieveBackStructure(data, f, field, id);
        return
    end
end

%--------------------------
% get model (NOMINAL)
fem=data.Model.Nominal;
%--------------------------

if opt(4) % project Pm on geometry
    if isfield(f, 'Pm')
        for i=1:size(f.Pm,1)
            Pmi=f.Pm(i,:);
            [Pmp, ~, flag]=point2PointNormalProjection(fem, Pmi, f.Master, f.SearchDist(1));
            if flag
                f.Pm(i,:)=Pmp;
                f.PmReset(i,:)=Pmp;
            end
        end
    end
end

%---------------
% update placement of part features
if strcmp(field,'Stitch') || strcmp(field,'Hole') || strcmp(field,'Slot')
    if opt(3)
        T0w=data.Input.Part(f.Master).Placement.T;

        % Pm...
        f.Pm=apply4x4(f.Pm,T0w(1:3,1:3), T0w(1:3,4)');

        % Nm...
        f.Nm=apply4x4(f.Nm,T0w(1:3,1:3), [0 0 0]);

        % Nt...
        f.Nt=apply4x4(f.Nt,T0w(1:3,1:3), [0 0 0]);
    end
end

% STEP 1: count points
    % nworkpoint: no. of points used for "work" calculation
    % nmodelpoint: no. of points used to define the "model"
if strcmp(field,'NcBlock') || strcmp(field,'ClampS') || strcmp(field,'ClampM') 
    nmodelpoint=1;
elseif strcmp(field,'Stitch')
    if f.Type{1}==1 || f.Type{1}==4 % linear; edge
        nmodelpoint=2;
    elseif f.Type{1}==2 || f.Type{1}==3 % circular; rigid link
       nmodelpoint=1;
    end
else
    nmodelpoint=1;
end

% STEP 2: update geometry and settings
f.Pam=[];
f.Nam=[];
f.Tam=[];
f.Vam=[];
f.Pas=[];
f.Nas=[];
npara=zeros(1,nmodelpoint);
for imodelppoint=1:nmodelpoint
    
    if strcmp(field,'NcBlock') || strcmp(field,'ClampS') || strcmp(field,'ClampM') 

        stype=f.Geometry.Shape.Type{1};
        if f.Geometry.Type{1}==1 % 1-point model
            if stype==1 || stype==2 || stype==3
                nworkpoint=1;
            elseif stype==4 || stype==5
                nworkpoint=1:2;
            end
        elseif f.Geometry.Type{1}==2 % 5-point model
            
            if stype==1 || stype==2 || stype==3
                nworkpoint=1:5;
            elseif stype==4 || stype==5
                nworkpoint=1:10;
            end
        end
    elseif strcmp(field,'Stitch')
            nworkpoint=imodelppoint;
    else
        nworkpoint=1;
    end

    % calculate local reference frame
    [R0c, flag]=localGetParametricFrame(f, fem, imodelppoint, field); 

    if ~flag
        ex=3;

        for i=1:length(f.Status)
          f.Status{i}=ex;
        end
        data=retrieveBackStructure(data, f, field, id);
        return
    end

    paratype=f.Parametrisation.Geometry.Type{imodelppoint}{1};
    
    % save rotation
    f.Parametrisation.Geometry.R{imodelppoint}=R0c;
   
    % calculate parameters
    if paratype==1 % reference
        t=0.0; v=0.0; n=0.0;
    else
        [t, v, n]=localExtractParameters(f, paratype, imodelppoint);
    end

    npara(imodelppoint)=length(t);
    
    % extract model and working points
    Pm=f.Pm;
    if strcmp(field,'NcBlock') || strcmp(field,'ClampS') || strcmp(field,'ClampM') 
        Pm=Pm(imodelppoint, :);
        Pm=size2Point(f, R0c, Pm);
    end
       
    % loop over all points
    for j=nworkpoint

        Pam=zeros(npara(imodelppoint),3);
        Nam=zeros(npara(imodelppoint),3);
        Tam=zeros(npara(imodelppoint),3);
        Vam=zeros(npara(imodelppoint),3);
        
        Pas=zeros(npara(imodelppoint),3);
        Nas=zeros(npara(imodelppoint),3);

        ex=zeros(1, npara(imodelppoint));

        % run over no. of parameters
        Pmj=Pm(j, :);
        for i=1:npara(imodelppoint)
            
            Pmi= Pmj + t(i)*R0c(:,1)' + v(i)*R0c(:,2)'; % translation along T and V
            
            % get points and normals
            [Pmpi, Nmpi, Tmpi, Vmpi, Pspi, Nspi, exi]=localgetProjectionPointNormal(f, fem, Pmi, imodelppoint, field);
                        
            ex(i)=exi;
               
            % add translation along N
            if strcmp(field,'ClampM')
                Pmpi=Pmpi + n(i)*Nmpi;
                Pspi=Pspi - n(i)*Nspi;
            else
                Pmpi=Pmpi + n(i)*Nmpi;
                Pspi=Pspi + n(i)*Nspi;
            end

            Pam(i,:)=Pmpi;
            Nam(i,:)=Nmpi;
            Tam(i,:)=Tmpi;
            Vam(i,:)=Vmpi;
            
            Pas(i,:)=Pspi;
            Nas(i,:)=Nspi;

        end
        
            % save out
            f.Status{j}=ex;

            f.Pam{j}=Pam;
            f.Nam{j}=Nam;
            
            %--
            if ~strcmp(field,'Stitch')
                f.Tam{j}=Tam;
                f.Vam{j}=Vam;
            else
                if f.Type{1}~=4 % edge
                    f.Tam{j}=Tam;
                    f.Vam{j}=Vam;
                end
            end
            
            f.Pas{j}=Pas;
            f.Nas{j}=Nas; 
                        
    end
        
end
%
%---------------------
% check gap condition for "stitch"
if strcmp(field,'Stitch')
    if f.Type{1}==3 % rigid link
        for geomparaidi=1:npara(imodelppoint)
            part_to_part_gap=norm(f.Pam{1}(geomparaidi,:)-f.Pas{1}(geomparaidi,:));
            if part_to_part_gap > f.Gap
                f.EnableReset=false;
            end
        end
    end
end
%
% save back
data=retrieveBackStructure(data, f, field, id);
                

% get local frame
function [T, V, flag]=getlocalframe(tT, N, typent)

%--
eps=1e-6;

% 
flag=false;
T=[];
V=[];

if typent==2 % model
    
    V=getY2X(N)';
    flag=true;
    
elseif typent==1 % user
    
    %--
    tV=cross(N, tT);

    l=norm(tV);

    if l<=eps
        return
    end

    V=tV/l;
    flag=true;

end

if flag
    T=cross(V, N);
end


%--
function [Pmp, Nmp, Tmp, Vmp, Psp, Nsp, ex]=localgetProjectionPointNormal(f, fem, P0, imodelpoint, field)

eps=1e-6;

% master
Pmp=[0 0 0]; % point
Nmp=[0 0 0]; % normal
Tmp=[0 0 0]; % tangent (T)
Vmp=[0 0 0]; % tangent (V)

% slave
Nsp=[0 0 0];
Psp=[0 0 0];

% flag
ex=0;

% check geometry projection
idpart=f.Master;

if strcmp(field,'CustomConstraint') || strcmp(field,'CustomConstraint') || strcmp(field,'NcBlock') || strcmp(field,'ClampS') || strcmp(field,'ClampM') || strcmp(field,'Stitch')  || strcmp(field,'Hole') || strcmp(field,'Slot') 

    N0=f.Nm(imodelpoint,:);

    if f.NormalType{1}==1 % user
        l=norm(N0);

        if l<=eps
            ex=3;
            return
        end

        N0=N0/l; % make sure the vector is normalised

       [Pmp, flag]=pointNormal2PointProjection(fem, P0, N0, idpart);
       Nmp=N0;
    elseif f.NormalType{1}==2 % model
       [Pmp, Nmp, flag]=point2PointNormalProjection(fem, P0, idpart, f.SearchDist(1));
    end

    % check for normal flipping
    if f.FlipNormal
        Nmp=-Nmp;
    end
    
end

if ~flag
  ex=1;
  return
end

% type of tangent (user/model)
typent=f.TangentType{1};
tT=f.Nt(imodelpoint, :);
[Tmp, Vmp, flag]=getlocalframe(tT, Nmp, typent);

if ~flag
  ex=1;
  return
end


% get slave data
if strcmp(field,'ClampM') || strcmp(field,'Stitch') 
    
    idpart=f.Slave;
     
    if f.NormalType{1}==1 % user
       [Psp, flag]=pointNormal2PointProjection(fem, P0, N0, idpart);
    elseif f.NormalType{1}==2 %  model
       [Psp, flag]=point2PointProjection(fem, P0, idpart, f.SearchDist(1));
    end
        
    Nsp=-Nmp;
    
    if ~flag
       ex=2;
       return
    end

end


% calculate points based on model (1-5 points)and shape
function Pm=size2Point(f, R0c, P0)

% NOTICE: P0 is already on the surface (no projection is needed!)

Pm=[];

% external rotation around Z axis 
ro=f.Geometry.Shape.Rotation*pi/180;

% build rotation around Z
R0z=[cos(ro) -sin(ro) 0
     sin(ro) cos(ro)  0
     0       0        1];

R0c=R0c*R0z;

% geometry type (1-5 points)
type=f.Geometry.Type{1};
% geometry shape
stype=f.Geometry.Shape.Type{1};
 
%-------------------------------
% STEP 2: get points
if type==1 % point % 1-point scheme
   if stype==1 || stype==2 || stype==3
        Pm=P0;   
      elseif stype==4 || stype==5
        Pitch=f.Geometry.Shape.Parameter.Pitch;
        P0c=zeros(2,3);
        P0c(1,1)=P0c(1,1)-Pitch/2;
        P0c(2,1)=P0c(2,1)+Pitch/2;  
        
        Pm=apply4x4(P0c, R0c, P0);
   end
elseif type==2 % sgeom % 5-point scheme
     
    if stype==1 % cylinder

        A=f.Geometry.Shape.Parameter.D;
        
        % 4-point
        P=[A/2 0 0
           0 A/2 0
           -A/2 0 0
           0 -A/2 0];

    elseif stype==2 || stype==3 % block or L-shape

        % size
        A=f.Geometry.Shape.Parameter.A;
        B=f.Geometry.Shape.Parameter.B;
        
       % 4-point
        P=[A/2 B/2 0
           -A/2 B/2 0
           -A/2 -B/2 0
           A/2 -B/2 0];
       
    elseif stype==4 % couple cylinder
        
        A=f.Geometry.Shape.Parameter.D;
        Pitch=f.Geometry.Shape.Parameter.Pitch;
       
        P=[A/2-Pitch/2 0 0
           -Pitch/2 A/2 0
           -A/2-Pitch/2 0 0
           -Pitch/2 -A/2 0
           A/2+Pitch/2 0 0
           Pitch/2 A/2 0
           -A/2+Pitch/2 0 0
           Pitch/2 -A/2 0];
       
        P0c=zeros(2,3);
        P0c(1,1)=P0c(1,1)-Pitch/2;
        P0c(2,1)=P0c(2,1)+Pitch/2;
       
    elseif stype==5 % couple prisma

        % size
        A=f.Geometry.Shape.Parameter.A;
        B=f.Geometry.Shape.Parameter.B;
        Pitch=f.Geometry.Shape.Parameter.Pitch;
        
        P=[A/2-Pitch/2 B/2 0
           -A/2-Pitch/2 B/2 0
           -A/2-Pitch/2 -B/2 0
           A/2-Pitch/2 -B/2 0
           A/2+Pitch/2 B/2 0
           -A/2+Pitch/2 B/2 0
           -A/2+Pitch/2 -B/2 0
           A/2+Pitch/2 -B/2 0 ];
       
        P0c=zeros(2,3);
        P0c(1,1)=P0c(1,1)-Pitch/2;
        P0c(2,1)=P0c(2,1)+Pitch/2;
       
    end

    % transform in global frame
    if stype==1 || stype==2 || stype==3
        P=apply4x4(P, R0c, P0);

        % add Pm to the list
        Pm=[P0
            P]; 
    elseif stype==4 || stype==5
        P=apply4x4(P, R0c, P0);
        P0c=apply4x4(P0c, R0c, P0);

        % add Pm to the list
        Pm=[P0c
            P]; 
    end

    
end


% calculate rotation matrix
function [R0c, flag]=localGetParametricFrame(f, fem, imodelpoint, field)

flag=true;
R0c=[];

P0=f.Pm(imodelpoint,:); 

[~, N, T, V, ~, ~, ex]=localgetProjectionPointNormal(f, fem, P0, imodelpoint, field);

if ex~=0
    flag=false;
    return
end

%... then, rotation matrix
R0c=[T', V', N']; 

if strcmp(field,'Slot')
    angle=f.Geometry.Shape.Rotation;
    angle=angle*pi/180;
    R0c=RodriguesRot(N, angle)*R0c;
end


%----------
function [t, v, n]=localExtractParameters(f, paratype, imodelpoint)

t=0;
v=0;
n=0;

if paratype==2 || paratype==13 % T
    t=f.Parametrisation.Geometry.T{imodelpoint};
    v=zeros(1,length(t));
    n=zeros(1,length(t));
elseif paratype==3 || paratype==14 % V
    v=f.Parametrisation.Geometry.V{imodelpoint};
    t=zeros(1,length(v));
    n=zeros(1,length(v));
elseif paratype==4 || paratype==15 % N
    n=f.Parametrisation.Geometry.N{imodelpoint};
    t=zeros(1,length(n));
    v=zeros(1,length(n));
elseif paratype==5 || paratype==16 % TV
    t=f.Parametrisation.Geometry.T{imodelpoint};
    v=f.Parametrisation.Geometry.V{imodelpoint};
    n=zeros(1,length(t));
elseif paratype==6 || paratype==17 % TN
    t=f.Parametrisation.Geometry.T{imodelpoint};
    n=f.Parametrisation.Geometry.N{imodelpoint};
    v=zeros(1,length(t));
elseif paratype==7 || paratype==18 % VN
    v=f.Parametrisation.Geometry.V{imodelpoint};
    n=f.Parametrisation.Geometry.N{imodelpoint};
    t=zeros(1,length(v));
elseif paratype==8 || paratype==19 % TVN
    t=f.Parametrisation.Geometry.T{imodelpoint};
    v=f.Parametrisation.Geometry.V{imodelpoint};
    n=f.Parametrisation.Geometry.N{imodelpoint};
end



