% Solve contact problem by using the "Penalty method"
function fem=femSolvePenalty(fem)

tic;

% initial values
flag=false;
countiter=1;

% max iteration allowed
maxiter=fem.Options.Solver.MaxIter;

maxpenalty=fem.Options.Solver.PenaltyStiffness;

Nscp=length(fem.Boundary.Constraint.ContactMPC);

%...
if Nscp>0 % at least one contact pair detected
    
    while true
        
            if fem.Options.Solver.PenaltyAdaptive % use adaptive method
                fem.Options.Solver.PenaltyStiffness=define_adaptive_penalty(maxpenalty, countiter);
            end
        
            fprintf('Non-Linear Iteration: %d\n', countiter);
            
            % PHASE 1: SOLVE LINEAR PROBLEM           
            disp('Solving Linear System')
            fem=femLinearSolve(fem);
                        
            % PHASE 2: check gap
            [fem, flagg]=checkGap(fem);
                    
            %...
            if fem.Sol.nLCt==0 % no active set
                flag=true;
                
                disp('Convergent solution: no contact was detected!')
                break
            end
            
            fprintf('=> minimum gap: %f - maximum Traction: %f\n',fem.Sol.Log.MinGap, fem.Sol.Log.MaxLag*fem.Options.Solver.PenaltyStiffness);
           
            fprintf('=> active pairs: %g\n',fem.Sol.nLCt);
            fprintf('=> penalty coefficient: %e\n',fem.Options.Solver.PenaltyStiffness);

            % check convergence
            if flagg 
                flag=true;
                
                disp('Convergent solution: contact was detected!')
                break
            end
                                      
            countiter=countiter+1;

            % max. iterations reached
            if countiter>=maxiter
                warning('Solution not found: maximum # of iterations reached!')
                break
            end

            disp('---')
            
    end
    
else
    
    
    % SOLVE LINEAR           
    disp('Solving Linear System...')
    fem=femLinearSolve(fem);
    
    flag=true;
    
    fem.Sol.Log.MinGap=0;
    fem.Sol.Log.MaxLag=0;
            
end

% save
if fem.Options.Solver.CheckTol %% check numerical residuals
    if norm(fem.Sol.res)>=fem.Options.Solver.EpsCheck
        warning('Residuals are greater than allowed tolerance! Default solution has been set to zero!') %#ok<WNTAG>
        fem.Sol.U(:)=0;
    end
end

fem.Sol.Log.Iter=countiter;
fem.Sol.Log.Done=flag;

fem.Sol.Log.TimeSolve=toc;

disp('---')
fprintf('Simulation time: %f seconds\n',fem.Sol.Log.TimeSolve);


% look for active constraints (gi<0)
function [fem,flag]=checkGap(fem)

% convergence criteria:
% 1. actual working set: gi>=0
% 2. previous working set: gi<0 (all pairs belonging to the previous working set are in compression)

% NOTICE: the pressure load is equal to F=penValue*g. Since "penValue" is
% always positive, to check the pressure it is enought to see the sign of
% "gi" (this means that all penalty stiffs are in compression state)


    
flagI=true; 
flagII=true;

% congergence check
flag=true;

% num. tolerance allowed
eps=fem.Options.Solver.Eps;

fmax=fem.Options.Min;

%---------------
% STEP 1: check previous working set
if ~isempty(fem.Boundary.Constraint.ContactWset)
    
    nc=length(fem.Boundary.Constraint.ContactWset);
    
    for i=1:nc
        
        id=fem.Boundary.Constraint.ContactWset(i).Id;
        u=fem.Sol.U(id);

        gi=fem.Boundary.Constraint.ContactWset(i).Gap; 

        coeff=fem.Boundary.Constraint.ContactWset(i).Coefficient;

        gi=gi + dot( u, coeff );
        
        % this pair is in traction mode (gi>0)
        if gi>0.0
            flagI=false;
        end
        
        if gi>=fmax
            fmax=gi;
        end
    
    end % end "i"
    
    
end

%---------------
% STEP 2: store actual working set
fem.Boundary.Constraint.ContactWset=[];

fem.Sol.nLCt=0;

gmin=fem.Options.Max;

Nscp=length(fem.Boundary.Constraint.ContactMPC);

tcount=0;
for i=1:Nscp

    id=fem.Boundary.Constraint.ContactMPC(i).Id;
    u=fem.Sol.U(id);
    
    gi=fem.Boundary.Constraint.ContactMPC(i).Gap; 

    coeff=fem.Boundary.Constraint.ContactMPC(i).Coefficient;

    gi=gi + dot( u, coeff );
    
    % --------------------------
    % save actual gap
    fem.Boundary.Constraint.ContactMPC(i).GapActual=gi;
    fem.Boundary.Constraint.ContactMPC(i).ValueActual=-gi;
    % --------------------------
    
    if gi < 0.0 % active set 

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
        tcount = tcount + length(fem.Boundary.Constraint.ContactWset(countLag).Coefficient)^2;

    end
    
    % get min gi
    if gi<=gmin
        gmin=gi;
    end

end

% check gap condition (gi>=0)
if gmin<=-eps %... I am allowing a bit penetration
    flagII=false;
end

if flagI==false || flagII==false
    flag=false;
end

% save back
fem.Sol.Log.MinGap=gmin;
fem.Sol.Log.MaxLag=fmax;

fem.Sol.Kast.n=fem.Sol.Kast.nreset+tcount;
           
