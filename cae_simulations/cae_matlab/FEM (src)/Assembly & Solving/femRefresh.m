% update "fem" structure including boundary conditions and domain properties
function fem=femRefresh(fem)

% fem: fem structure

%----------------------------------
% define domain conditions
disp('Refreshing FEM equations...')

% set initial not-zero-numbers
%--------------------------------
fem.Sol.Kast.n=fem.Sol.Kast.ndofs;
%--------------------------------

% define boundary conditions (bilateral - node) - SPC
fem=setBilateralConstraintNode(fem);

% define rigid links - MPC
fem=setRigidLinkConstraint(fem);

% set pin-hole or pin-slot constraint - MPC
fem=setPinHoleConstraint(fem);
fem=setPinSlotConstraint(fem); 

% define boundary conditions (bilateral - element) - MPC
fem=setBilateralConstraintElement(fem);
  
% define contact pairs - UNILATERAL
fem=setContactConstraint(fem);
fem=subSampleContactPairs(fem); % run sub-sampling
cpoint=length(fem.Boundary.Constraint.ContactMPC);

% define dimples - UNILATERAL
fem=setDimpleConstraint(fem);
dpoint=length(fem.Boundary.Constraint.ContactMPC)-cpoint;

% define unilateral constraint - UNILATERAL
fem=setUnilateralConstraint(fem);
upoint=length(fem.Boundary.Constraint.ContactMPC)-cpoint-dpoint;

% define load at domain side
fem=setDomainLoad(fem);

% define force at node side
fem=setLoadNode(fem);

% define force at element side
fem=setLoadElement(fem);

% define user-defined loads
fem=setLoadUser(fem);

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% get no. of MPCs
fem.Sol.nLMPC=length(fem.Boundary.Constraint.MPC); % # of Lagrange multiplier for MP
% get no. of SPCs
fem.Sol.nLSPC=length(fem.Boundary.Constraint.SPC.Id);
%
fem.Sol.nLCt=0; % # of Lagrange multiplier for contact ("0" at beginning)

%-----------------------------------
% read # of dofs
ndof=fem.Sol.nDoF;
nSpc=fem.Sol.nLSPC; 
nMpc=fem.Sol.nLMPC; 
nCt=fem.Sol.nLCt; 

nload=length(fem.Boundary.Load.Value);

% solver type:
tsolver=fem.Options.Solver.Method;

% tot. # of dofs
if strcmp(tsolver,'lagrange')
    
    nTot=ndof+nSpc+nMpc+nCt; % augmented matrix
elseif strcmp(tsolver,'penalty')
    
    nTot=ndof; % work just on primary variables
else
    
    error('FEMP (Refresh equations): Constraint handling method not recognised!') 
end
%-----------------------

% set initial values
fem.Sol.U=zeros(nTot,1); % dofs

fem.Sol.res=zeros(nTot,1); % residuals

fem.Sol.R=zeros(nTot,1); % reaction forces

% set gap variables
nctpairs=length(fem.Boundary.ContactPair);
nnode=size(fem.xMesh.Node.Coordinate,1);

if nctpairs>0
    for i=1:nctpairs
       fem.Sol.Gap(i).Gap=zeros(1,nnode);    %ones(1,nnode)*fem.Options.Max; % gap for each contact pair
       
       fem.Sol.Gap(i).max=fem.Options.Min;
       fem.Sol.Gap(i).min=fem.Options.Max;  
    end
end

% disp log
disp('---')
disp('Compiled Equations:')

fprintf('    Bilateral Constraints: \n');
fprintf('    No. of Degrees of Freedom: %i\n',ndof);
fprintf('    No. of Single-Point-Constraints: %i\n',nSpc);
fprintf('    No. of Multi-Point-Constraints: %i\n',nMpc);
disp('---')
fprintf('    Unilateral Constraints: \n');
fprintf('    No. of Contact Points: %i\n',cpoint);
fprintf('    No. of Dimple Points: %i\n',dpoint);
fprintf('    No. of Unilateral Points: %i\n',upoint);
disp('---')
fprintf('    Load Conditions: \n');
fprintf('    No. of Load Points: %i\n',nload);


%--------------------------------------------------------------------------

%-------------------
function fem=subSampleContactPairs(fem)

% subs: [0, 1]

np=length(fem.Boundary.ContactPair);

% run checks...
if np==0
    return
end

if isempty(fem.Boundary.Constraint.ContactMPC)
    return
end

% run sub-sampling
idpairs=[fem.Boundary.Constraint.ContactMPC(:).IdContactPair];
idlist=1:length(idpairs);

idpairsubs=[];
for i=1:np

    subs=fem.Boundary.ContactPair(i).Sampling;
    
    ii=idlist(idpairs==i);

    nc=length(ii);

    % random selection
    sel = randperm(nc);

    % subs percentage
    sel = sel(1:floor(nc*subs));    
    ii=ii(sel);

    % update
    idpairsubs=[idpairsubs, ii]; %#ok<AGROW>

end

% save back
fem.Boundary.Constraint.ContactMPC=fem.Boundary.Constraint.ContactMPC(idpairsubs);

