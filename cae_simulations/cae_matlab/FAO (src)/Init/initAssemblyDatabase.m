% initialise solution structure
function d=initAssemblyDatabase()
   
d.Solver.Method={1,'penalty','lagrange'};
d.Solver.PenaltyStiffness=1e9; % penalty stiffness
d.Solver.PenaltyAdaptive=false; % use adaptive method
d.Solver.LinearSolver={2,'umfpack','cholmod','cholmodGPU','bs','pardiso'};

d.Solver.MaxIter=100; % max. iterations allowed
d.Solver.Eps=1e-2; % numerical tolerance allowed

d.Solver.UsePreLoad=false; % pre-load option (true/false)
d.Solver.UseSubModel=false; % sub-model option (true/false)
d.Solver.UseParallel=false; % use parallel cluster (true/false)
d.Solver.UseSolution=false; % reset solution (true/false)
d.Solver.UseSoftSpring=false; % use soft spring (true/false)
d.Solver.SoftSpring=1e-1; % soft spring stiffness

d.Solver.PolChaos.UsePolChaos=false; % use pol chaos (true/false)
d.Solver.PolChaos.Degree=2; % degree of polynomial chaos expansion
d.Solver.PolChaos.RatioSample=1; % oversampling of min. number of pol chaos coefficient
d.Solver.PolChaos.MaxIter=100; % max no. of generations (when using MC)
d.Solver.PolChaos.PopulationSize=1000; % no. of samples to compute stochastic outputs

d.Min=-1e9; % minimum value allowed
d.Max=1e9; % maximum value allowed
d.Eps=1e-3; % numerical tolerance allowed

d.PolChaos.Csi=[]; % stochastic parameters
d.PolChaos.A=[]; % weight matrix for pol chaos expansion
d.PolChaos.nPC=1; % no. of pol chaos samples
d.PolChaos.nSip=[]; % no of stochastic paramters per part
d.PolChaos.nSi=0; % total no of stochastic paramters 

d.Parameter=cell(0,7); % list of parameters
d.Group=[1,0,1,2]; % [id, min, max, resolution]
d.X.Value=[]; % parameter table
d.X.Status=[]; % parameter table (status)
    % 0: failed to compute
    % 1: computed
d.X.nD=1; % no. of samples
d.X.ID=1; % sample ID
d.SamplingStrategy={1, 'full factorial', 'random', 'user'}; % sampling strategy
d.SamplingOptions.SampleSize=50; % sample size (only for SamplingStrategy="random")
d.SamplingOptions.IdTable=1; % ID of the parameter table (only for SamplingStrategy="user")

% Solution (U)
d.U{1}=[]; % solution vector
d.GAP{1}=[]; % solution gap vector
d.Log{1}=[]; % log section

d.SubModel.Node=[]; % node ids 
d.SubModel.DoF=[]; % DoF
d.SubModel.U=[]; % [u, v, w, alfa, beta, gamma] displacement field

% Pre-load condition
d.PreLoad.DoF=[]; 
d.PreLoad.F=[]; % [u, v, w, alfa, beta, gamma] force field 
