% Run assembly simulation
function varargout=simulationCore(data, sdata, idpart, opt)
%
% Inputs
% data: data structure
% sdata: list of selection (true/false)
    % They refer to data.Input.Selection
    % If "sdata" is empty => use all nodes in the model
% idpart: part list
% opt: 
    % "refreshAll" => refresh existing geometry + update stiffness matrix (default option)
    % "refresh" => refresh existing geometry without updating stiffness matrix
%
% Outputs
% (1) data: updated data structure
% (2) computationalTime- seconds (optional)
% (3) simulationFlag - true/false => computed/failed to compute (optional)
% (4) Ka: assembled stiffnes matrix - before applying any constraints (optional)
%
%------------------------------
% log file
filelog='log_sol.txt';
fclose('all');
%------------------------------

if nargin==3
    opt='refreshAll';
end

% get ideal model
femc=data.Model.Nominal;
node=femc.xMesh.Node.Coordinate;

% define options for assembly
if strcmp(opt, 'refreshAll')
    femc.Options.StiffnessUpdate=true;
else
    femc.Options.StiffnessUpdate=false;
end
femc.Options.MassUpdate=false;
femc.Options.ConnectivityUpdate=false; 

%-------------------------------------
femc.Options.UseActiveSelection=true; 
%-------------------------------------

% define solver settings
t=data.Assembly.Solver.Method{1};
femc.Options.Solver.Method=data.Assembly.Solver.Method{t+1}; 
femc.Options.Solver.PenaltyStiffness=data.Assembly.Solver.PenaltyStiffness; 

femc.Options.Solver.UseSoftSpring=data.Assembly.Solver.UseSoftSpring;
femc.Options.Solver.SoftSpring=data.Assembly.Solver.SoftSpring;

t=data.Assembly.Solver.LinearSolver{1};
femc.Options.Solver.LinearSolver=data.Assembly.Solver.LinearSolver{t+1};

femc.Options.Solver.MaxIter=data.Assembly.Solver.MaxIter;
femc.Options.Solver.Eps=data.Assembly.Solver.Eps;

femc.Options.Solver.PenaltyAdaptive=data.Assembly.Solver.PenaltyAdaptive; 

femc.Options.Eps=data.Assembly.Eps;
femc.Options.Max=data.Assembly.Max;
femc.Options.Min=data.Assembly.Min;

femc.Options.GapFrame='def'; % use deformed frame for gap calculation

useparallel=data.Assembly.Solver.UseParallel;

%--------------------------------------------------------------------------------

% STEP 0: get active nodes (per each part)
[seleNodesbyParts, ~]=getLocalActiveSelection(data, sdata, idpart);

%--
np=length(idpart);

% STEP 1: build current assembly
countnodes=0;
activeNode=[];
for i=1:np
    geom=data.Input.Part(idpart(i)).Geometry.Type{1};
    
    if geom>1 % NOT IDEAL GEOMETRY
        
        ppart=data.Input.Part(idpart(i)).Geometry.Parameter;
        
        idnode=femc.Domain(idpart(i)).Node;
        uvw=data.Input.Part(idpart(i)).D{ppart};
        
        if isempty(uvw)
            error('Simulation core - attempting to create non-ideal geometry from empty data set!')
        end
        % update coordinates
        node(idnode,:)=node(idnode,:)+uvw;
        femc.xMesh.Node.Coordinate=node;
        
    end
    
    activeNode(i).Node=seleNodesbyParts{i};
    activeNode(i).Status=data.Input.Part(idpart(i)).Enable;
    activeNode(i).Part=idpart(i);
    
    countnodes=countnodes+length(seleNodesbyParts{i});
end

%--
if countnodes==0
    varargout{1}=data;

    if nargout==2
       varargout{1}=data;
       varargout{2}=0;
    end

    if nargout==3
        varargout{1}=data;
        varargout{2}=[];
        varargout{3}=false;
    end
    
    if nargout==4
        varargout{1}=data;
        varargout{2}=[];
        varargout{3}=false;
        varargout{4}=[];
    end
    
    return
end

%--
femc=femPreProcessing(femc, activeNode);

% loop over all parameters
npara=data.Assembly.X.nD;

% run CALCULATION
ndofs=data.Model.Nominal.Sol.nDoF;
Utot=zeros(ndofs, npara);
gaptot=cell(1, npara);
logtot=cell(1, npara);

t_start=tic;
if useparallel % USE PARALLEL MODE
    parfor geomparaid=1:npara
        [Utot(:, geomparaid), gaptot{geomparaid}, logtot{geomparaid}]=run_local_sim(data, femc, geomparaid);
    end
else % USE SEQUENTIAL MODE
    for geomparaid=1:npara
        [Utot(:, geomparaid), gaptot{geomparaid}, logtot{geomparaid}]=run_local_sim(data, femc, geomparaid);    
    end
end
t_solve=toc(t_start);

%-------------
if isempty(data.Assembly.U{1})
    nsol=0;
else
    nsol=length(data.Assembly.U);
end

for geomparaid=1:npara
    
    nsol=nsol+1; % solution counter
    
    U=Utot(:, geomparaid);
    gap=gaptot{geomparaid};
    log=logtot{geomparaid};
    
    % STEP 5: store solution
    for i=1:np
        idnode=data.Model.Nominal.Domain(idpart(i)).Node;
        
        iddofs=data.Model.Nominal.xMesh.Node.NodeIndex(idnode,:);
                
        geom=data.Input.Part(idpart(i)).Geometry.Type{1};
    
        if geom>1 % NOT IDEAL
              
            ppart=data.Input.Part(idpart(i)).Geometry.Parameter;
            
            uvw=data.Input.Part(idpart(i)).D{ppart};
            
            % update solution set by deformation field
            U(iddofs(:,1)) = U(iddofs(:,1)) + uvw(:,1); % U
            U(iddofs(:,2)) = U(iddofs(:,2)) + uvw(:,2); % V
            U(iddofs(:,3)) = U(iddofs(:,3)) + uvw(:,3); % W
        
        end
        
        data.Input.Part(idpart(i)).U{nsol}=[U(iddofs(:,1)),...
                                                  U(iddofs(:,2)),...
                                                  U(iddofs(:,3)),...
                                                  U(iddofs(:,4)),...
                                                  U(iddofs(:,5)),...
                                                  U(iddofs(:,6))]; % [U, V, W, Rx, Ry, Rz]
    end
    
    % store all data set
    data.Assembly.U{nsol}=U;
    data.Assembly.GAP{nsol}=gap;
    data.Assembly.Log{nsol}=log;
        
    % write log file
    idlog=fopen(filelog,'a');
    fprintf(idlog,'      Design configuration [%g]\r\n',geomparaid);
    
    if log.Done
        fprintf(idlog,'            Status: solved\r\n');
    else
        fprintf(idlog,'            Status: NOT solved\r\n');
    end
    fprintf(idlog,'            No. of iterations: %g\r\n',log.Iter);
    fclose(idlog); 
    
end

% write log file
idlog=fopen(filelog,'a');
fprintf(idlog,'    Simulation time [sec]: %f\r\n',t_solve);
fclose(idlog);

% save out...
varargout{1}=data;

if nargout==2
   varargout{1}=data;
   varargout{2}=t_solve;
end

if nargout==3
    varargout{1}=data;
    varargout{2}=t_solve;
    varargout{3}=true;
end

if nargout==4
    varargout{1}=data;
    varargout{2}=t_solve;
    varargout{3}=true;
    varargout{4}=femAssemblyStiffness(femc); % Save the assembly stiffness matrix (if required)
end

%--------------------------------------------

%--------
function [U, gap, log]=run_local_sim(data, femc, geomparaid)

U=zeros(data.Model.Nominal.Sol.nDoF,1);

% STEP 1: set fixtures
femc=fixtureModeling(data, femc, geomparaid);

% STEP 2: refresh equations
femc=femReset(femc);    
femc=femRefresh(femc);

% STEP 3: solve model
femc=femSolve(femc);

% STEP 4: reshape solution vector
Uc=femc.Sol.U;

%------------
active_nodes=femc.Selection.Node.Active;
iddofs=data.Model.Nominal.xMesh.Node.NodeIndex(active_nodes,:);
iddofs=sort(iddofs(:));

% save outputs
U(iddofs)=Uc; 

gap=femc.Sol.Gap;
log=femc.Sol.Log;

