% Solve contact problem by using an iterative approach (Active set Method) based on Lagrange Multiplier ("active set" method)
function fem=femSolveLagrange(fem)

tic;

% initial values
flag=false;
countiter=1;

% num. tolerance allowed
eps=fem.Options.Solver.Eps;

% max iteration allowed
maxiter=fem.Options.Solver.MaxIter;

Nscp=length(fem.Boundary.Constraint.ContactMPC);

%...
if Nscp>0 % at least one contact pair detected
    
    while true
        
            sr=sprintf('Non-Linear Iteration: %d', countiter);
            disp(sr)
            
            flagl=false; % lagrange check
            flagg=false; % gap check
            
            % PHASE 1: SOLVE LINEAR PROBLEM 1           
            disp('Solving Linear System: STEP 1...')
            fem=femLinearSolve(fem);
            
            % PHASE 2: ADDING PHASE (gi<=0)
            [fem,ncoeff]=checkConditionI(fem);
            
            %...
            if fem.Sol.nLCt==0 % no active set
                flag=true;
                disp('Convergent solution: no contact was detected!')
                break
            end
            
            if fem.Sol.Log.MinGap>=-eps % I am allowing a bit penetration
                flagg=true;
            end

            % PHASE 3: SOLVE LINEAR PROBLEM 2            
            disp('Solving Linear System: STEP 2...')
            fem=femLinearSolve(fem);
             
            % PHASE 4: REMOVING PHASE (lamda>0)
            [fem,flagl]=checkConditionII(fem,flagl,ncoeff);

            sr=sprintf('=> minimum gap: %f - maximum Lagrange: %f',fem.Sol.Log.MinGap, fem.Sol.Log.MaxLag);
            disp(sr)
            
            sr=sprintf('=> active pairs: %g',fem.Sol.nLCt);
            disp(sr)
            
            % check convergence
            if flagg && flagl
                flag=true;
                disp('Convergent solution: contact was detected!')
                break
            end
                                                  
        countiter=countiter+1;
        
        % max. iterations reached
        if countiter>=maxiter
            warning('Solution not found: maximum # of iterations reached!') %#ok
            break
        end
        
        disp('---')

    end
    
else
    
    % SOLVE LINEAR           
    disp('Solving Linear System...')
    fem=femLinearSolve(fem);
            
end

% save
if fem.Options.Solver.CheckTol %% check numerical residuals
    if norm(fem.Sol.res)>=fem.Options.Solver.EpsCheck
        warning('Residuals are greater than allowed tolerance! Default solution was set to zero!') %#ok<WNTAG>
        fem.Sol.U(:)=0;
    end
end

fem.Sol.Log.Iter=countiter;
fem.Sol.Log.Done=flag;

fem.Sol.Log.TimeSolve=toc;

disp('---')
fprintf('Simulation time: %f seconds\n',fem.Sol.Log.TimeSolve);


% look for active constraints (gi<=0)
function [fem,ncoeff]=checkConditionI(fem)

fem.Boundary.Constraint.ContactWset=[];
ncoeff=[];

fem.Sol.nLCt=0;

gmin=fem.Options.Max;

% num. tolerance allowed
eps=fem.Options.Solver.Eps;

Nscp=length(fem.Boundary.Constraint.ContactMPC);

for i=1:Nscp

    id=fem.Boundary.Constraint.ContactMPC(i).Id;
    u=fem.Sol.U(id);
    
    gi=fem.Boundary.Constraint.ContactMPC(i).Gap; 

    coeff=fem.Boundary.Constraint.ContactMPC(i).Coefficient;

    gi=gi + dot( u, coeff );
    if gi<=eps % active set

        % create related MPC
        fem.Sol.nLCt=fem.Sol.nLCt+1;
        
        countLag=fem.Sol.nLCt;

        % ADD to working set
        fem.Boundary.Constraint.ContactWset(countLag).Id=fem.Boundary.Constraint.ContactMPC(i).Id;
        fem.Boundary.Constraint.ContactWset(countLag).Value=fem.Boundary.Constraint.ContactMPC(i).Value;

        %
        fem.Boundary.Constraint.ContactWset(countLag).Coefficient=fem.Boundary.Constraint.ContactMPC(i).Coefficient;

        %
        fem.Boundary.Constraint.ContactWset(countLag).IdContactPair=fem.Boundary.Constraint.ContactMPC(i).IdContactPair;
       
        %
        fem.Boundary.Constraint.ContactWset(countLag).Gap=fem.Boundary.Constraint.ContactMPC(i).Gap;

        %
        fem.Boundary.Constraint.ContactWset(countLag).Psl=fem.Boundary.Constraint.ContactMPC(i).Psl;
        fem.Boundary.Constraint.ContactWset(countLag).Pms=fem.Boundary.Constraint.ContactMPC(i).Pms;

        %
        fem.Boundary.Constraint.ContactWset(countLag).Type=fem.Boundary.Constraint.ContactMPC(i).Type;
        
        %
        fem.Boundary.Constraint.ContactWset(countLag).Reaction.Id=i;
        
        % update # of not-zero entries (ADD)
        ncoeff(countLag)=length(fem.Boundary.Constraint.ContactWset(countLag).Coefficient)*2;

    end

    % get min gi
    if gi<=gmin
        gmin=gi;
    end

end

% save
fem.Sol.Log.MinGap=gmin;
fem.Sol.Kast.n=fem.Sol.Kast.nreset + sum( ncoeff );


% look for inactive constraints (lag. Multiplier greater than or equal to 0)
function [fem,flagl]=checkConditionII(fem,flagl,ncoeff)

% num. tolerance allowed
eps=fem.Options.Solver.Eps;

% read DoFs
ndof=fem.Sol.nDoF;
nSpc=fem.Sol.nLSPC; 
nMpc=fem.Sol.nLMPC; 
    
fem.Sol.Lamda=fem.Sol.U(ndof+nSpc+nMpc+1:end);
idlag=find(fem.Sol.Lamda>=eps);

if isempty(idlag)
    flagl=true;
    
    % save
    fem.Sol.Log.MaxLag=full(max(fem.Sol.Lamda));
else
    fem.Sol.nLCt=fem.Sol.nLCt-length(idlag);
    fem.Boundary.Constraint.ContactWset(idlag)=[];
            
    % save
    fem.Sol.Log.MaxLag=full(max(fem.Sol.Lamda));
    
    % update # of not-zero entries (REMOVE)
    fem.Sol.Kast.n=fem.Sol.Kast.n - sum( ncoeff(idlag) );
end


            