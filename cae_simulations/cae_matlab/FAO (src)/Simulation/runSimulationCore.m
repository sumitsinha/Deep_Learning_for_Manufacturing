% Master function running assembly simulation
function varargout=runSimulationCore(data, sdata, pdata, refreshOpt)

% INPUT
% data: data structure
% sdata: list of selection (true/false)
    % They refer to data.Input.Selection
    % If "sdata" is empty => use all nodes in the model
% pdata: part data {part ID, geometry, simulation mode, parameter ID}
    % part ID: part ID
    % geometry: 1="nominal"; 2="morphed"; 3="measured"
    % simulation mode: 1="deterministic"; 2="stochastic"
    % parameter ID: parameter ID used to generated "measured" part
% refreshOpt: 
    % "refreshAll" => refresh existing geometry + update stiffness matrix (default option)
    % "refresh" => refresh existing geometry without updating stiffness matrix

    
% OUTPUT
% (1) data: updated data structure
% (2) flagsim: true/false => computed/failed to compute 
% (3) Ka: assembled stiffnes matrix - before applying any constraints (optional)

%------------------------------
% log file
filelog='log_sol.txt';
fclose('all');
%------------------------------

if nargin==3
    refreshOpt='refreshAll';
end

if isempty(pdata)
    varargout{1}=data;

    if nargout==2
       varargout{1}=data;
       varargout{2}=false;
    end
    
    if nargout==3
       varargout{1}=data;
       varargout{2}=false;
       varargout{3}=[];
    end
    return
end

idparts=pdata(:,1)'; % list of parts
tparts=pdata(:,3); % simulation mode ('Deterministic'/'Stochastic')

% STEP 1: generate polynomial chaos
[data, flag]=generatePolChaosExpasion(data, idparts, tparts);

t_solve=0;
if ~flag % run deterministic solution
  fprintf('Running deterministic solution...\n');
  
  % STEP 2: generate variational geometry
  opt.Flag='deterministic';
  data=runSimulationCoreVariationGeometry(data, idparts, opt);
  
  % STEP 3: run simulation
  if nargout==3
      [data, t_solve, flagsim, Ka]=simulationCore(data, sdata, idparts, refreshOpt);
  else
      [data, t_solve, flagsim]=simulationCore(data, sdata, idparts, refreshOpt);
  end
  
  if ~flagsim
    varargout{1}=data;

    if nargout==2
       varargout{1}=data;
       varargout{2}=flagsim;
    end
    
    if nargout==3
       varargout{1}=data;
       varargout{2}=flagsim;
       varargout{3}=Ka;
    end
    return
  end
    
else % run stochastic solution
  fprintf('Running stochastic solution...\n');  
  
  opt.Flag='stochastic';
  
  if data.Assembly.Solver.PolChaos.UsePolChaos
     nPC=data.Assembly.PolChaos.nPC;
  else
     nPC=data.Assembly.Solver.PolChaos.MaxIter;
  end
  for i=1:nPC
      
      % write log file
      idlog=fopen(filelog,'a');
      fprintf(idlog,'Stochastic sample [%g]\r\n',i);
      fclose(idlog); 
    
      opt.Parameter=i; % pol chaos parameter
      
      % STEP 2: generate variational geometry
      data=runSimulationCoreVariationGeometry(data, idparts, opt);

      % STEP 3: run simulation
     if nargout==3
          [data, t_solvei, flagsim, Ka]=simulationCore(data, sdata, idparts, refreshOpt);
      else
          [data, t_solvei, flagsim]=simulationCore(data, sdata, idparts, refreshOpt);
      end

      if ~flagsim
        varargout{1}=data;

        if nargout==2
           varargout{1}=data;
           varargout{2}=flagsim;
        end

        if nargout==3
           varargout{1}=data;
           varargout{2}=flagsim;
           varargout{3}=Ka;
        end
        return
      end
  
      t_solve=t_solve+t_solvei;
       
  end
    
end

% write log file
idlog=fopen(filelog,'a');
fprintf(idlog,'Total Simulation time [sec]: %f\r\n',t_solve);
fclose(idlog);

% save back
varargout{1}=data;
if nargout==2
   varargout{1}=data;
   varargout{2}=flagsim;
end
if nargout==3
   varargout{1}=data;
   varargout{2}=flagsim;
   varargout{3}=Ka;
end

