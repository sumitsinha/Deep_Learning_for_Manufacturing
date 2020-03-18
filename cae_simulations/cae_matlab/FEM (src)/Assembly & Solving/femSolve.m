% solve fem model
function fem=femSolve(fem)

tsolver=fem.Options.Solver.Method;

%-----------------------------------
% not-zero-number saved for back-up
fem.Sol.Kast.nreset=fem.Sol.Kast.n;
%-----------------------------------
    
if strcmp(tsolver,'lagrange')
    
    disp('---')
    disp('Solving model - Lagrangian multiplier method')
    disp('---')
    
    fem=femSolveLagrange(fem);
    
elseif strcmp(tsolver,'penalty')
    
    disp('---')
    disp('Solving model - Penalty method')
    disp('---')
    
    fem=femSolvePenalty(fem);
    
else
    
    error('FEMP (Linear Solver): Constraint handling method not recognised!') 
end

%-------------------------------------------------------------------------
disp('---')

% Transform all data into global coordinate frame...
fem=femGlobalSolution(fem);

% Calculating deformed frame...
fem=getDeformedFrame(fem);

% Calculate Gap Variables...
fem=getGapsVariable(fem);




