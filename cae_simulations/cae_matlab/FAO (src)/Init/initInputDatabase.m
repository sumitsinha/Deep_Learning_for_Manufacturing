% initialise input data structure
function d=initInputDatabase(flag)

%-----------------------------
% NOTICE:
    % .Master/Slave => part ID used for geometrial projection and calculation of reference frames
    % .DomainM/DomainS => part ID used for simulation in the FEM kernel
    % .Status: -1/0/1/2/3 = not updated/success/failed to project on master/failed to project on slave/wrong input
%-----------------------------

%-----------------------------
if strcmp(flag,'Parameter')
    
   d.Name='parameter';
   d.X=0.0; % data
   d.Input={'X1'}; % name of variables

elseif strcmp(flag,'Selection')
    
   d.Name='selection';   
   d.Pm=[0 0 0]; % [X Y Z] (centroid) 
   d.PmReset=[0 0 0]; % [X Y Z] (centroid) 
   d.Nm1=[1 0 0]; % [Nx Ny Nz] (main direction of the selection)
   d.Nm2=[0 1 0]; % [Nx Ny Nz] (secondary direction of the selection)
   d.Rm=[50 50 50]; % size of selection
   d.Type={2,'Prisma', 'Ellipsoid'}; % type
   
   d.Graphic.Color='c';
   d.Graphic.EdgeColor='k';
   d.Graphic.FaceAlpha=0.5;
   d.Graphic.Show=false;
   d.Graphic.ShowEdge=false;
    
elseif strcmp(flag,'PartG')
    
   d.Name='partG';
   d.Mesh={''};
   d.Status=-1;

   d.Graphic.Color=[0 0 0];
   d.Graphic.EdgeColor='k';
   d.Graphic.FaceAlpha=1.0;
   
   d.Graphic.Show=true;   
   d.Graphic.ShowEdge=false;
   
   d.Domain=[];
       
elseif strcmp(flag,'Part')
    
   d.Name='part';
   d.Enable=true;
   d.EnableReset=true;
   d.Selection.Type={1,'Automatic', 'Selection'}; % if "Automatic" => all nodes belonging to that part
   d.Selection.Node=[]; % node ids in active selection
   d.E=210e3; % Young modulus
   d.nu=0.33; % Poission ratio
   d.Th=1.0; % Material thickness
   d.FlexStatus=true; % true=deformable part; false=rigid part
   d.Geometry.Type={1, 'Nominal', 'Morphed', 'Measured'};
   d.Geometry.Parameter=1; % ParameterID to select dev pattern from d.D{ParameterID} in case d.Geometry.Type=="Measured"
   d.Geometry.Mode={1, 'Deterministic', 'Stochastic'};
   d.FlipNormal=false; % flip normal
   d.CoP={''}; % cloud of point file used to build ".Geometry.Type=>Measured"
   d.Mesh={''}; % mesh file
   d.Morphing=initMorphingMesh; % morphing mesh settings
   d.Offset=0.0; % for deviation calculation using cloud of points data
   d.SearchDist=[10.0 5.0]; % normal and tangential searching distances
   d.Status=-1;
         
   d.Placement.T=eye(4,4); % actual 4x4 placement matrix
   d.Placement.TStore{1}=eye(4,4); % actual 4x4 placement matrix (archive)
   d.Placement.UCS=eye(4,4); % User coordinate system
   d.Placement.UCSStore{1}=eye(4,4); % User coordinate system (archive)
   d.Placement.UCSreset=eye(4,4); % User coordinate system
   d.Placement.ShowFrame=false; % true/false => show/hide part UCS
   
   d.Parametrisation.Type=[0, 0, 0, 0, 0, 0]; % ['alfa', 'beta', 'gamma', 'T', 'V', 'N']
   d.Parametrisation.Name={'alfa', 'beta', 'gamma', 'T', 'V', 'N'};
   d.Parametrisation.UCS=0; % 0/1 => global/local UCS
   
   d.D{1}=[]; % store deviation vectors (u, v, w)
   d.U{1}=[]; % store solution vectors (u, v, w, alfa, beta, gamma)
   
   d.Graphic.Color='g';
   d.Graphic.EdgeColor='k';
   d.Graphic.FaceAlpha=1.0;
   
   d.Graphic.Show=true;
   d.Graphic.ShowNormal=false;
   d.Graphic.ShowCoP=false;
   
   d.Graphic.ShowEdge=false;
   
elseif strcmp(flag,'Robot')
   
   d.Name='robot';
   d.Enable=true;
   d.EnableReset=true;
   d.Model={1, 'ABB6700-235', 'ABB6700-200', 'ABB6620'};
   d.Tool.Model={1, 'WeldMaster', 'WLS400A'};
   
   d.Parameter=[];
   d.JointLim=[];
   d.SpeedLim=[];
   d.Kinematics=[];
   d.Geom=[];
   
   d.Options=[];
   
   d.Graphic=[];
   
   d.Status{1}=-1;
             
elseif strcmp(flag,'Stitch')
   
    d.Name='stitch';
    d.Enable=true;
    d.EnableReset=true;
    d.Type={4, 'Linear', 'Circular', 'Spot', 'Edge'}; 
    d.Resolution=10; % no. of points to be created 
    d.Master=0;
    d.Slave=0;
    d.DomainM=0;
    d.DomainS=0;
    d.Pm=[0 0 0
          0 0 0
          0 0 0]; % start/end/via point
    d.PmReset=[0 0 0
              0 0 0
              0 0 0];
    d.Nm=[0 0 1
          0 0 1]; % normal of start/end point
    d.NmReset=[0 0 1
               0 0 1];
    d.FlipNormal=true; % flip normal
    d.Nt=[1 0 0
          1 0 0]; % tangent
    d.NtReset=[1 0 0
                1 0 0]; % tangent of start/end point
    d.Diameter=8.0;
    d.Gap=0.0; % part to part gap => only if part to part gap >=.Gap, then the stitch is added to the model
    d.NormalType={2, 'User', 'Model'}; % "model"; "user"
    d.TangentType={2, 'User', 'Model'}; % "model"; "user"
    d.SearchDist=[5.0 5.0]; % normal and tangential
    
    d.Parametrisation.Geometry.Type{1}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN',[],[],[],...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF',[],[],[]}; 
    d.Parametrisation.Geometry.Type{2}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN',[],[],[],...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF',[],[],[]};
    
    d.Parametrisation.Geometry.T{1}=0;
    d.Parametrisation.Geometry.V{1}=0;
    d.Parametrisation.Geometry.N{1}=0;
    d.Parametrisation.Geometry.T{2}=0;
    d.Parametrisation.Geometry.V{2}=0;
    d.Parametrisation.Geometry.N{2}=0;
    
    d.Parametrisation.Geometry.R{1}=eye(3,3);
    d.Parametrisation.Geometry.R{2}=eye(3,3);
    
    d.Parametrisation.Geometry.ShowFrame=false;
    
    d.Parametrisation.DoF{1}=[];
    d.Parametrisation.DoF{2}=[];
    
    %--------
    d.Graphic.Color='m';
    d.Graphic.EdgeColor='k';
    d.Graphic.FaceAlpha=1.0;
    d.Graphic.Show=true;
    d.Graphic.ShowNormal=false;
    d.Graphic.ShowManipulator=false;
    d.Graphic.ShowEdge=false;
    
    d.Constraint.Ds=1; % distance from starting point
    d.Constraint.De=1; % distance from ending point
    d.Constraint.Dc=1; % cylindrical zone
        
    %--
    d.Knot=[]; % interpolated points (used by "Edge" option)
    d.Pam=[];
    d.Nam=[];
    d.Tam=[];
    d.Vam=[];
    d.Pas=[];
    d.Nas=[];
    d.Status={-1, -1};
    
elseif strcmp(flag,'Dimple')
    
    d.Name='dimple';
    d.Enable=true;
    d.EnableReset=true;
    d.Master=0;
    d.Slave=0;
    d.Pm=[0 0 0];
    d.Height=0.2;
    d.Stiffness=0.0;
    d.Offset=0.0;
    d.MasterFlip=false;
    d.Length=3.0;
    d.SearchDist=[10.0 5]; % normal and tangential
    
    d.Graphic.Color='k';
    d.Graphic.EdgeColor='k';
    d.Graphic.FaceAlpha=1.0;
    d.Graphic.Show=true;
    d.Graphic.ShowNormal=false;
    d.Graphic.ShowEdge=true;

    d.Constraint.Ds=1; % distance from starting point
    d.Constraint.De=1; % distance from ending point
    d.Constraint.Dc=1; % cylindrical zone
    
    d.Pam=[];
    d.Nam=[];
    d.Tam=[];
    d.Vam=[];
    d.Pas=[];
    d.Nas=[];
    
    d.StitchId=0;
    
    d.Status{1}=-1;
        
elseif strcmp(flag,'Hole') || strcmp(flag,'Slot') 
        
    d.Name='holeslot';
    d.Enable=true;
    d.EnableReset=true;
    d.DomainM=0;
    d.Master=0;
    d.Pm=[0 0 0];
    d.PmReset=[0 0 0];
    d.Nm=[0 0 1];
    d.NmReset=[0 0 1];
    d.FlipNormal=false; % flip normal
    d.Nt=[0 1 0];
    d.NtReset=[0 1 0];
    d.NormalType={1, 'User', 'Model'}; % "model"; "user"
    d.TangentType={1, 'User', 'Model'}; % "model"; "user"
    d.SearchDist=[10.0 5]; % normal and tangential
    
    % if "Hole" => DoC=[u, v] into local reference system (R)
    % if "Slot" => DoC=[u] into local reference system (R)
    if strcmp(flag,'Hole')
        d.Parametrisation.Geometry.Type{1}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN','u','v','uv',...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF','u ON/OFF','v ON/OFF','uv ON/OFF'};
        d.Parametrisation.DoC.u{1}=0; % degree of constraint
        d.Parametrisation.DoC.v{1}=0; % degree of constraint
    elseif strcmp(flag,'Slot')
         d.Parametrisation.Geometry.Type{1}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN','u','v','uv',...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF','u ON/OFF','v ON/OFF','uv ON/OFF'};
        d.Parametrisation.DoC.u{1}=0; % degree of constraint
    end
    
    d.Parametrisation.Geometry.T{1}=0;
    d.Parametrisation.Geometry.V{1}=0;
    d.Parametrisation.Geometry.N{1}=0;

    d.Parametrisation.Geometry.R{1}=eye(3,3);
    
    d.Parametrisation.Geometry.ShowFrame=false;
    
    d.Geometry.Shape.Rotation=0.0; % deg 
    d.Geometry.Shape.Parameter.Diameter=20; % mm
    d.Geometry.Shape.Parameter.Length=40; % % length of slot

    d.Graphic.Color='m';
    d.Graphic.EdgeColor='k';
    d.Graphic.FaceAlpha=1.0;
    d.Graphic.Show=true;
    d.Graphic.ShowNormal=false;
    d.Graphic.ShowEdge=false;
    
    d.Constraint.Ds=1; % distance from starting point
    d.Constraint.De=1; % distance from ending point
    d.Constraint.Dc=1; % cylindrical zone

    d.Pam=[];
    d.Nam=[];
    d.Tam=[];
    d.Vam=[];
    d.Pas=[];
    d.Nas=[];

    d.Status{1}=-1;
  
elseif strcmp(flag,'CustomConstraint')
        
    d.Name='CustomConstraint';
    d.Enable=true;
    d.EnableReset=true;
    d.DomainM=0;
    d.DomainS=0;
    d.Master=0;
    d.Pm=[0 0 0];
    d.PmReset=[0 0 0];
    d.Nm=[0 0 1];
    d.NmReset=[0 0 1];
    d.FlipNormal=false; % flip normal
    d.NormalType={2, 'User', 'Model'}; % "model"; "user"
    d.Nt=[0 0 1];
    d.NtReset=[0 0 1];
    d.TangentType={2, 'User', 'Model'}; % "model"; "user"
    %
    d.DoFs=true(1,6); % Degrees of Freedom
    d.Value=zeros(1,6); % Value of constraint
    d.Type={1,'bilateralCartesian','bilateralVectorTra','unilateralLock','unilateralFree'};
    %
    d.Parametrisation.Geometry.Type{1}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN',[],[],[],...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF',[],[],[]};
    d.Parametrisation.Geometry.T{1}=0;
    d.Parametrisation.Geometry.V{1}=0;
    d.Parametrisation.Geometry.N{1}=0;
    d.Parametrisation.Geometry.R{1}=eye(3,3);
    
    d.Parametrisation.Geometry.ShowFrame=false;
    
    d.Parametrisation.DoF{1}=[0 0 0 0 0 0];

    d.SearchDist=[10.0 5]; % normal and tangential

    d.Graphic.Color='c';
    d.Graphic.EdgeColor='k';
    d.Graphic.FaceAlpha=1.0;
    d.Graphic.Show=true;
    d.Graphic.ShowNormal=false;
    d.Graphic.ShowEdge=true;
    
    d.Pam=[];
    d.Nam=[];
    d.Tam=[];
    d.Vam=[];
    d.Pas=[];
    d.Nas=[];

    d.Status{1}=-1;

elseif strcmp(flag,'Locator')
    
    d.Name='locator';
    d.Enable=true;
    d.EnableReset=true;
    d.DomainM=0;
    d.DomainS=0;
    d.Master=0;
    d.Slave=0;
    d.Pm=[0 0 0];
    d.PmReset=[0 0 0];
    d.Nm=[0 0 1];
    d.NmReset=[0 0 1];
    d.FlipNormal=false; % flip normal
    d.Nt=[0 1 0];
    d.NtReset=[0 1 0];
    d.SearchDist=[5.0 5]; % normal and tangential
    d.NormalType={2, 'User', 'Model'}; % "model"; "user"
    d.TangentType={2, 'User', 'Model'}; % "model"; "user"

    d.Parametrisation.Geometry.Type{1}={1, 'Reference', 'T', 'V', 'N', 'TV', 'TN', 'VN', 'TVN',[],[],[],...
                                                         'ON/OFF','T ON/OFF', 'V ON/OFF', 'N ON/OFF', 'TV ON/OFF', 'TN ON/OFF', 'VN ON/OFF', 'TVN ON/OFF',[],[],[]};
    d.Parametrisation.Geometry.T{1}=0;
    d.Parametrisation.Geometry.V{1}=0;
    d.Parametrisation.Geometry.N{1}=0;
    d.Parametrisation.Geometry.R{1}=eye(3,3);
    
    d.Parametrisation.Geometry.ShowFrame=false;
    
    d.Parametrisation.DoF{1}=[];

    d.Geometry.Type={2, '1-Point', '5-Point'}; % "point"; "sgeom"
    d.Geometry.Shape.Type={1, 'Cylinder', 'Prisma', 'L-Shape', 'Couple Cylinder', 'Couple Prisma'}; 
    d.Geometry.Shape.Parameter.D=20.0; % diameter
    d.Geometry.Shape.Parameter.A=20.0; % length of prisma/L-shape
    d.Geometry.Shape.Parameter.B=20.0; % width of prisma/L-shape
    d.Geometry.Shape.Parameter.C=20.0; % length of locator
    d.Geometry.Shape.Parameter.L=40.0; % length of L-shape
    d.Geometry.Shape.Parameter.Pitch=60.0; % pitch distance between 2 pins (couple pins)
    d.Geometry.Shape.Rotation=0.0; % rotation around axis [deg]
    
    d.Graphic.Color='c';
    d.Graphic.EdgeColor='k';
    d.Graphic.FaceAlpha=1.0;
    d.Graphic.Show=true;
    d.Graphic.ShowNormal=false;
    d.Graphic.ShowEdge=false;

    d.Pam=[];
    d.Nam=[];
    d.Tam=[];
    d.Vam=[];
    d.Pas=[];
    d.Nas=[];

    d.Status{1}=-1;
      
elseif strcmp(flag,'Contact')
    
    d.Name='contact';
    d.Enable=true; 
    d.EnableReset=true;
    d.Use=true; % if ".Use==true" => avoid penetration
    d.Master=0;
    d.Slave=0;
    d.MasterFlip=false;
    d.Offset=0.0;
    d.SearchDist=[10.0 15.0]; % normal distance and sharpe angle
    d.Sampling=1; % sampling rate

    d.Status{1}=-1;
            
end

