% initialize the "fem" structure
function varargout=femInit(wdir)

% wdir: full path pointing to "Source File" folder
%--------------------------------------------------------------------------

if nargin~=0
     addpath(genpath(wdir));
end


%---------------------------------------------------
% just install FEMP folders
if nargout==0
    return
end

% GENERAL OPTIONS
fem.Options.Physics='structure'; % "heat";...
fem.Options.Solver.Type='stationary'; % "stationary"; "transient"; ...

fem.Options.Solver.TimeSolver.Mode='backwardEuler';
fem.Options.Solver.TimeSolver.Times.Min=0;
fem.Options.Solver.TimeSolver.Times.Max=1;
fem.Options.Solver.TimeSolver.Times.Dt=0.1;
fem.Options.Solver.TimeSolver.Times.N=100;

fem.Options.Solver.Method='penalty'; % "penalty" or "lagrange"
fem.Options.Solver.PenaltyStiffness=1e10; % penalty stiffness
fem.Options.Solver.PenaltyAdaptive=false; % use adaptive method
fem.Options.Solver.LinearSolver='cholmod'; % "umfpack", "cholmod", "cholmodGPU", "bs", "pardiso"
fem.Options.Solver.StoreAssembly=false; % true/false => store assembled stiffness matrix

fem.Options.Solver.UseSoftSpring=false; % use soft spring (true/false)
fem.Options.Solver.SoftSpring=1e-1; % sfot spring stiffness

fem.Options.Solver.MaxIter=100; % max. iterations allowed
fem.Options.Solver.Eps=1e-8; % numerical tolerance allowed
fem.Options.Solver.EpsCheck=1e-6; % numerical tolerance allowed
fem.Options.Solver.CheckTol=false;

fem.Options.Eps=1e-6; % numerical tolerance allowed
fem.Options.Max=1e9; % maximum value allowed
fem.Options.Min=-1e9; % minimum value allowed

fem.Options.GapFrame='def';% frame used to calculate the gap distibution ("ref", "def")

fem.Options.StiffnessUpdate=true; % when "false" stiffness matrix is not updated/computed
fem.Options.MassUpdate=false; % when "false" mass matrix is not updated/computed
fem.Options.UseActiveSelection=false; % when "true" only active elements are considered (based on the current selection... "fem.Selection"). If false then use ALL elements
fem.Options.ConnectivityUpdate=true; % when "false" connectivity matrices for mesh denoising are not updated/computed

% MESH AND GEOMETRY

% extended mesh
fem.xMesh.Ucs=[];
fem.xMesh.Reference=[]; % false when not transformed/updated

fem.xMesh.Element(1)=struct();
fem.xMesh.Element(1).Type=[]; % bar; beam; tria; quad; hexa; tetra
fem.xMesh.Element(1).Element=[]; % node connections
fem.xMesh.Element(1).Component=[]; % component identification

% mat. properties
fem.xMesh.Element(1).Material.E=[]; % Young
fem.xMesh.Element(1).Material.ni=[]; % Poisson
fem.xMesh.Element(1).Material.lamda=[]; % Shear correction factor
fem.xMesh.Element(1).Material.Density=[]; % Density

fem.xMesh.Element(1).Constant.Th=[]; % thickness

% matrices
fem.xMesh.Element(1).ElementIndex=[];
fem.xMesh.Element(1).ElementNodeIndex=[];
fem.xMesh.Element(1).Tmatrix.T0lDoF=[]; 
fem.xMesh.Element(1).Tmatrix.T0lGeom=[]; % 3x3 rotation matrix
fem.xMesh.Element(1).Tmatrix.P0lGeom=[]; % origin of local coordinate frame
fem.xMesh.Element(1).Tmatrix.T0ucs=[];
fem.xMesh.Element(1).Tmatrix.Normal=[];
fem.xMesh.Element(1).Tmatrix.NormalReset=[];

fem.xMesh.Element(1).Ke=[]; % stiffness matrix
fem.xMesh.Element(1).Me=[]; % mass matrix

%.............

fem.xMesh.Node.Coordinate=[];
fem.xMesh.Node.CoordinateReset=[];
fem.xMesh.Node.Tnode=[];   % trasformation matrix for node
fem.xMesh.Node.Component=[];
fem.xMesh.Node.NodeIndex=[];

fem.xMesh.Node.Normal=[];
fem.xMesh.Node.NormalReset=[];
    
fem.Geometry.Ucs=[];

% FILTER OPTIONS

%--------------------------------------------------------------------------
fem.Denoise.Options.MaxIter=3;
fem.Denoise.Options.Domain=1; % list of domains to be snoothed

fem.Denoise.Tria=[];
fem.Denoise.Domain(1).Tria=[];

fem.Denoise.Trianormal=[];
fem.Denoise.Connectivity.Node2Ele=[];
fem.Denoise.Connectivity.Ele2Ele=[];


% DCT OPTIONS

%--------------------------------------------------------------------------

fem.Dct.Domain=1; % domain id
fem.Dct.Option.Energy=0.9; % energy compaction
fem.Dct.Option.CorrThr=0.2; % correlation limit to keep
fem.Dct.Option.VoxX=100;
fem.Dct.Option.VoxY=100;
fem.Dct.Option.VoxZ=100;
fem.Dct.Option.VoxelPercentage = 0.5; % percentage for no. of voxels
fem.Dct.Option.VoxelSelection = 1; % 1: for voxel size input, 2: for the voxel percentage input
fem.Dct.Option.ScaleBB=0.05; % percentage of bb scale
fem.Dct.Option.SearchDist=10; % searching distance
fem.Dct.Option.Offset=0.0; % searching distance

fem.Dct.Option.LaplaceInterp=true; % true/false = enable/disable
fem.Dct.Option.WeightCorrection=true; % true/false = enable/disable

fem.Dct.Option.EnergyCoeffManual=true; % use manual selection for energy compaction
fem.Dct.Option.NoEnergyCoeff=10; % Number of Energy Coefficients to keep when NoEnergyCoeffSelection is "true"

fem.Dct.Option.CorrCoeffManual=true;
fem.Dct.Option.NoCorrCoeff=10; % Number of Correlated Coefficients to keep when NoCorrCoeffSelection is "true"

% Mapping options

%--------------------------------------------------------------------------

fem.Mapping.Source=1; % source domain
fem.Mapping.Destination=1; % destination domain
fem.Mapping.SearchDist=10; % searching distance
fem.Mapping.MapVariable='u'; % v, w, alfa, beta, gamma, user

% BOUNDARY CONDITIONS
%--------------------------------------------------------------------------
% pin-hole constraint
% fem.Boundary.Constraint.PinHole(i).Pm=[]; % hole position
% fem.Boundary.Constraint.PinHole(i).Nm1=[]; % main axis
% fem.Boundary.Constraint.PinHole(i).Nm2=[]; % secondary axis
% fem.Boundary.Constraint.PinHole(i).Domain=[]; % domain
% fem.Boundary.Constraint.PinHole(i).SearchDist=[]; % search distance
% fem.Boundary.Constraint.PinHole(i).Value=[]; % constraint values
% fem.Boundary.Constraint.PinHole(i).UserExp=[];

%-
% fem.Boundary.Constraint.PinHole(i).ConstraintId=[]; % low-level constraint id

%--------------------------------------------------------------------------
% pin-slot constraint
% fem.Boundary.Constraint.PinSlot(i).Pm=[]; % slot position
% fem.Boundary.Constraint.PinSlot(i).Nm1=[]; % slot axis 1
% fem.Boundary.Constraint.PinSlot(i).Nm2=[]; % slot axis 2
% fem.Boundary.Constraint.PinSlot(i).Domain=[]; % domain
% fem.Boundary.Constraint.PinSlot(i).SearchDist=[]; % search distance
% fem.Boundary.Constraint.PinSlot(i).Value=[]; % constraint values
% fem.Boundary.Constraint.PinSlot(i).UserExp=[];

%-
% fem.Boundary.Constraint.PinSlot(i).ConstraintId=[]; % low-level constraint id

%--------------------------------------------------------------------------
% boundary constraints (bilateral constraints)
% fem.Boundary.Constraint.Bilateral.Node(i).Node=[];
% fem.Boundary.Constraint.Bilateral.Node(i).Reference=[]; %- cartesian, vectorTra, vectorRot
% fem.Boundary.Constraint.Bilateral.Node(i).Nm=[]; % given unit vector
% fem.Boundary.Constraint.Bilateral.Node(i).DoF=[];
% fem.Boundary.Constraint.Bilateral.Node(i).Value=[];
% fem.Boundary.Constraint.Bilateral.Node(i).UserExp=[];

%---
% fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Id=[]; spc or mpc identification
% fem.Boundary.Constraint.Bilateral.Node(i).Type=[]; % assigned / not assigned

%--------------------------------------------------------------------------
% element constraints (apply constraint anywhere in the geometry domain)
% fem.Boundary.Constraint.Bilateral.Element(i).Pm=[];
% fem.Boundary.Constraint.Bilateral.Element(i).Reference=[]; %- cartesian, vectorTra, vectorRot
% fem.Boundary.Constraint.Bilateral.Element(i).SearchDist=[]; % search distance
% fem.Boundary.Constraint.Bilateral.Element(i).Nm=[]; % given unit vector
% fem.Boundary.Constraint.Bilateral.Element(i).DoF=[];
% fem.Boundary.Constraint.Bilateral.Element(i).Value=[];
% fem.Boundary.Constraint.Bilateral.Element(i).Domain=[];
% fem.Boundary.Constraint.Bilateral.Element(i).UserExp=[];

%-
% fem.Boundary.Constraint.Bilateral.Element(i).Pms=[]; % domain point projection
% fem.Boundary.Constraint.Bilateral.Element(i).Type=[]; % node/element projection or not assigned

%---
% fem.Boundary.Constraint.Bilateral.Element(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.Constraint.Bilateral.Element(i).Reaction.Id=[]; spc or mpc identification

fem.Boundary.Constraint.PinHole=[];
fem.Boundary.Constraint.PinSlot=[];
fem.Boundary.Constraint.Bilateral.Node=[];
fem.Boundary.Constraint.Bilateral.Element=[];

% SPC conditions (only LINEAR constraints)
fem.Boundary.Constraint.SPC.Id=[]; % id dofs constrained
fem.Boundary.Constraint.SPC.Value=[];

%---------------------
fem.Boundary.Constraint.SPC.Reaction=[]; % reaction force

%--------------------------------------------------------------------------
% rigid links
% fem.Boundary.Constraint.RigidLink(i).Pm=[]; % point
% fem.Boundary.Constraint.RigidLink(i).Nm=[]; % vector
% fem.Boundary.Constraint.RigidLink(i).SearchDist=[]; % searching distance
% fem.Boundary.Constraint.RigidLink(i).Master=[]; % master component
% fem.Boundary.Constraint.RigidLink(i).Slave=[]; % slave component
% fem.Boundary.Constraint.RigidLink(i).UserExp=[];
% fem.Boundary.Constraint.RigidLink(i).Frame=[]; "ref" or "def" 

%-
% fem.Boundary.Constraint.RigidLink(i).Pms=[]; % master point
% fem.Boundary.Constraint.RigidLink(i).Psl=[]; % slave point
% fem.Boundary.Constraint.RigidLink(i).Type=[]; % status

%---
% fem.Boundary.Constraint.RigidLink(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.Constraint.RigidLink(i).Reaction.Id=[]; spc or mpc identification

fem.Boundary.Constraint.RigidLink=[];

%--------------------------------------------------------------------------
% unilateral constraints
% fem.Boundary.Constraint.Unilateral(i).Pm=[]; % point
% fem.Boundary.Constraint.Unilateral(i).SearchDist=[]; % search distance
% fem.Boundary.Constraint.Unilateral(i).Nm=[]; % normal direction
% fem.Boundary.Constraint.Unilateral(i).Size=[]; % true/false
% fem.Boundary.Constraint.Unilateral(i).Nt=[]; % tangent direction
% fem.Boundary.Constraint.Unilateral(i).Pmsize=[]; % true points 
% fem.Boundary.Constraint.Unilateral(i).SizeA=[]; % clamp size A (main axis)
% fem.Boundary.Constraint.Unilateral(i).SizeB=[]; % clamp size B (second axis) 
% fem.Boundary.Constraint.Unilateral(i).Offset=[]; % offset
% fem.Boundary.Constraint.Unilateral(i).Domain=[];
% fem.Boundary.Constraint.Unilateral(i).Constraint=[]; free / lock
% fem.Boundary.Constraint.Unilateral(i).UserExp=[];
% fem.Boundary.Constraint.Unilateral(i).Frame=[]; "ref"; "def"

%-
% fem.Boundary.Constraint.Unilateral(i).Pms=[]; % domain point projection
% fem.Boundary.Constraint.Unilateral(i).Type=[]; % status

%---
% fem.Boundary.Constraint.Unilateral(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.Constraint.Unilateral(i).Reaction.Id=[]; spc or mpc identification

fem.Boundary.Constraint.Unilateral=[];

% Contact pair conditions:
% fem.Boundary.ContactPair(i).Master=[]; % master component
% fem.Boundary.ContactPair(i).MasterFlip=[]; % flip master normal(true/false)
% fem.Boundary.ContactPair(i).Slave=[]; % slave component
% fem.Boundary.ContactPair(i).SearchDist=[]; % searching distance
% fem.Boundary.ContactPair(i).SharpAngle=[]; %--
% fem.Boundary.ContactPair(i).Offset=[]; % offset
% fem.Boundary.ContactPair(i).Enable=[]; % true/false
% fem.Boundary.ContactPair(i).Sampling=[]; % sampling value
% fem.Boundary.ContactPair(i).Frame=[]; "ref"; "def"

%-
% fem.Boundary.ContactPair(i).Pms=[];
% fem.Boundary.ContactPair(i).Psl=[];
% fem.Boundary.ContactPair(i).MasterId=[];
% fem.Boundary.ContactPair(i).SlaveId=[];
% fem.Boundary.ContactPair(i).Type=[];

%---
% fem.Boundary.ContactPair(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.ContactPair(i).Reaction.Id=[]; spc or mpc identification

fem.Boundary.ContactPair=[];

% Dimple pair conditions:
% fem.Boundary.DimplePair(i).Pm=[]; % dimple position
% fem.Boundary.DimplePair(i).Master=[]; % master component
% fem.Boundary.DimplePair(i).MasterFlip=[]; % flip master normal(true/false)
% fem.Boundary.DimplePair(i).Slave=[]; % slave component
% fem.Boundary.DimplePair(i).SearchDist=[]; % searching distance
% fem.Boundary.DimplePair(i).Offset=[]; % offset
% fem.Boundary.DimplePair(i).Height=[]; % dimple height
% fem.Boundary.DimplePair(i).Frame=[]; "ref"; "def"

%-
% fem.Boundary.DimplePair(i).Pms=[];
% fem.Boundary.DimplePair(i).Psl=[];
% fem.Boundary.DimplePair(i).Type=[]; %status

%---
% fem.Boundary.DimplePair(i).Reaction.Type='spc'/'mpc'/'unilateral'=1/2/3
% fem.Boundary.DimplePair(i).Reaction.Id=[]; spc or mpc identification

fem.Boundary.DimplePair=[];

%--------------------------------------------------------------------------
% MPC conditions (only LINEAR constraints)
%.........................................................................
% fem.Boundary.Constraint.MPC(i).Id=[]; % id dofs constrained
% fem.Boundary.Constraint.MPC(i).Coefficient=[]; % ... related coefficient
% fem.Boundary.Constraint.MPC(i).Value=[];

%---------------
% fem.Boundary.Constraint.MPC(i).Reaction=[]; % reaction force
fem.Boundary.Constraint.MPC=[];

%--------------------------------------------------------------------------
% MPC conditions for contact constraints
%.........................................................................
% fem.Boundary.Constraint.ContactMPC(i).Id=[]; % id dofs constrained
% fem.Boundary.Constraint.ContactMPC(i).Coefficient=[]; % ... related coefficient
% fem.Boundary.Constraint.ContactMPC(i).Value=[];
% fem.Boundary.Constraint.ContactMPC(i).IdContactPair=[]; "id" for contact pair; "-1" for unilateral, ....
% fem.Boundary.Constraint.ContactMPC(i).Gap=[]; ... signed distance
% fem.Boundary.Constraint.ContactMPC(i).Type=[]; node-to-surface; node-to-node
% fem.Boundary.Constraint.ContactMPC(i).Pms=[];
% fem.Boundary.Constraint.ContactMPC(i).Psl=[];
fem.Boundary.Constraint.ContactMPC=[];

%--------------------------------------------------------------------------
% MPC conditions for contact constraints solving (WORKING set)
%.........................................................................
% fem.Boundary.Constraint.ContactWset(i).Id=[]; % id dofs constrained
% fem.Boundary.Constraint.ContactWset(i).Coefficient=[]; % ... related coefficient
% fem.Boundary.Constraint.ContactWset(i).Value=[];
% fem.Boundary.Constraint.ContactWset(i).IdContactPair=[]; related contact pair id
% fem.Boundary.Constraint.ContactWset(i).Gap=[]; ... signed distance
% fem.Boundary.Constraint.ContactWset(i).Type=[]; node-to-surface; node-to-node
% fem.Boundary.Constraint.ContactWset(i).Pms=[];
% fem.Boundary.Constraint.ContactWset(i).Psl=[];
fem.Boundary.Constraint.ContactWset=[];

% boundary loads

% node
% fem.Boundary.Load.Node(1).Node=[];
% fem.Boundary.Load.Node(1).Reference=[]; %- cartesian, vectorTra, vectorRot
% fem.Boundary.Load.Node(1).Nm=[];
% fem.Boundary.Load.Node(1).DoF=[];
% fem.Boundary.Load.Node(1).Value=[];
fem.Boundary.Load.Node=[];

% element
% fem.Boundary.Load.Element(1).Pm=[];
% fem.Boundary.Load.Element(1).Reference=[]; %- cartesian, vectorTra, vectorRot
% fem.Boundary.Load.Element(1).SearchDist=[];
% fem.Boundary.Load.Element(1).Nm=[]; 
% fem.Boundary.Load.Element(1).DoF=[];
% fem.Boundary.Load.Element(1).Value=[];
% fem.Boundary.Load.Element(1).Domain=[];

% fem.Boundary.Load.Element(i).Pms=[]; % domain point projection
% fem.Boundary.Load.Element(i).Type=[]; % node/element projection or not assigned
fem.Boundary.Load.Element=[];

%--
% user
fem.Boundary.Load.User.Domain=[]; % it could have multiple domains
fem.Boundary.Load.User.Value=[];

%--
fem.Boundary.Load.DofId=[];
fem.Boundary.Load.Value=[];

%.............

% DOMAIN CONDITIONS
fem.Domain(1)=struct(); % 

fem.Domain(1).Status=true; % true/false

fem.Domain(1).Element=[]; % id elements
fem.Domain(1).Node=[]; % id nodes

fem.Domain(1).NormalFlip=false;

fem.Domain(1).Material.E=[]; % Young
fem.Domain(1).Material.ni=[]; % Poisson
fem.Domain(1).Material.lamda=[]; % Shear correction factor
fem.Domain(1).Material.Density=[]; % Density

fem.Domain(1).Constant.Th=[]; % thickness

fem.Domain(1).Load.Flag=false; % domain load flag (true/false)
fem.Domain(1).Load.Value=[]; % domain load

% Sub-modelling options
%--------------------------------------------------------------------------

fem.Domain(1).SubModel.CuttingSt=[]; % 3 nodes ids belonging to the cutting edge (for each component)
fem.Domain(1).SubModel.CuttingId=[]; % nodes ids belonging to the cutting edge (for each component)
fem.Domain(1).SubModel.SearchDist=10; % searching distance

% element selection
fem.Selection.Element.Status=[]; % true/false
fem.Selection.Element.Tria.Count=[]; % no. of trias
fem.Selection.Element.Tria.Id=[]; 
fem.Selection.Element.Tria.Element=[];

fem.Selection.Element.Quad.Count=[]; % no. quads
fem.Selection.Element.Quad.Id=[]; 
fem.Selection.Element.Quad.Element=[]; 

% node selection
fem.Selection.Node.Status=[]; % true/false
fem.Selection.Node.Active=0; % list of active nodes
fem.Selection.Node.Boundary=[]; % list of boundary nodes in the selected region
fem.Selection.Node.Count=[]; % no. of active nodes


% SOLUTION
fem.Sol.Node2Element=[]; % node to element connection
fem.Sol.Element2Element=[]; % % element to element connection

% total number of DoFs within the model
fem.Sol.nDoF=0; % # of DoF of the model
fem.Sol.nLSPC=0; % # of Lagrange multiplier for SPC (bilateral)
fem.Sol.nLMPC=0; % # of Lagrange multiplier for MPC (bilateral)
fem.Sol.nLCt=0; % # of Lagrange multiplier for contact (unilateral)
fem.Sol.nDom=0; % # domains

% LOG info
fem.Sol.Log.MinGap=[];
fem.Sol.Log.MaxLag=[];

fem.Sol.Log.TimeSolve=0;
fem.Sol.Log.Iter=0;
fem.Sol.Log.Done=false;

% TRIA
fem.Sol.Tria.Count=0; % # of tria
fem.Sol.Tria.Id=[];
fem.Sol.Tria.Element=[];

% QUAD
fem.Sol.Quad.Count=0; % # of quad
fem.Sol.Quad.Id=[];
fem.Sol.Quad.Element=[];

% Gauss points:

% QUAD Rule
% one point
fem.Sol.Gauss.Map.One.xg=0;
fem.Sol.Gauss.Map.One.wg=2;

% two points
fem.Sol.Gauss.Map.Two.xg=[1/sqrt(3) -1/sqrt(3)];
fem.Sol.Gauss.Map.Two.wg=[1 1];

% three points
fem.Sol.Gauss.Map.Three.xg=[0 sqrt(3/5) -sqrt(3/5)];
fem.Sol.Gauss.Map.Three.wg=[8/9 5/9 5/9];

% TRIA rules
% one point
fem.Sol.Gauss.Tria.One.xg=[1/3 1/3];
fem.Sol.Gauss.Tria.One.wg=1;

% two points
fem.Sol.Gauss.Tria.Two.xg=[1/2 0
                           0 1/2
                           1/2 1/2];
fem.Sol.Gauss.Tria.Two.wg=[1/3
                           1/3
                           1/3];

% three points
fem.Sol.Gauss.Tria.Three.xg=[1/3 1/3
                           1/5 3/5
                           1/5 1/5
                           3/5 1/5];
fem.Sol.Gauss.Tria.Three.wg=[-27/48
                           25/48
                           25/48
                           25/48];
                       
fem.Sol.Kast.Ka=[];
fem.Sol.Kast.Ma=[];
fem.Sol.Kast.n=0; % no. of not-zero-number
fem.Sol.Kast.ndofs=0; % no. of not-zero-number considering only stiffness items
fem.Sol.Kast.nreset=0; % no. of not-zero-number (reset purpose)
    
fem.Sol.U=[]; % all set
fem.Sol.res=[]; % numerical residuals


%--
% STORE DATA SET
fem.Sol.SolId=1; % solution ID

fem.Sol.USet.U=cell(1,1); % store data set
fem.Sol.USet.R=cell(1,1); % store data set

fem.Sol.ModelSet.Boundary=cell(1,1); % store model
fem.Sol.ModelSet.Domain=cell(1,1); % store model
%--------------------


%-- 
% user variables
fem.Sol.UserExp=[];

%--
% gap variables

%fem.Sol.Gap(1).Gap=[]; 
%fem.Sol.Gap(1).max=[];
%fem.Sol.Gap(1).min=[];
fem.Sol.Gap=[]; % actual gap distribution

%fem.Sol.GapFitUp(1).Gap=[]; 
%fem.Sol.GapFitUp(1).max=[];
%fem.Sol.GapFitUp(1).min=[];
fem.Sol.GapFitUp=[]; % gap distribution (as per CAD product). Use "getGapsVariableFitUp" to calculate this field

fem.Sol.GSet=cell(1,1); % store gap data set

fem.Sol.Lamda=[]; % contains just the reaction force for unilateral constraints
fem.Sol.RLamda=[]; % contains all reaction forces

% deformed frame
fem.Sol.DeformedFrame.Node.Coordinate=[]; 
fem.Sol.DeformedFrame.Node.Normal=[]; 
fem.Sol.DeformedFrame.Node.NormalReset=[]; 

fem.Sol.DeformedFrame.Element(1).Tmatrix.T0lGeom=[];
fem.Sol.DeformedFrame.Element(1).Tmatrix.P0lGeom=[];
fem.Sol.DeformedFrame.Element(1).Tmatrix.Normal=[];
fem.Sol.DeformedFrame.Element(1).Tmatrix.NormalReset=[];

% set initial post-processing options
fem=femPostInit(fem);

% save back
varargout{1}=fem;


