% get reaction forces
function fem=femReactionForce(fem)

disp('Recovering reaction forces...')

% solver type:
tsolver=fem.Options.Solver.Method;
    
%------------------------
% get SPC reaction forces
if strcmp(tsolver,'penalty')
    
    % get penalty value
    penval=fem.Options.Solver.PenaltyStiffness;
    
    id=fem.Boundary.Constraint.SPC.Id; % id dofs constrained
    val=fem.Boundary.Constraint.SPC.Value; % related values
    
    if ~isempty(id)
        u=fem.Sol.U(id);
        
        r=-getDiff(u, val) * penval; % "-" for reaction force
        
        % save
        fem.Boundary.Constraint.SPC.Reaction=r;
        
    end

elseif strcmp(tsolver,'lagrange')
    
    % read # of dofs
    nSpc=fem.Sol.nLSPC; 

    fem.Boundary.Constraint.SPC.Reaction=fem.Sol.RLamda(1:nSpc);
    
else
    
    error('FEMP (Reaction force processing): Constraint handling method not recognised!') 
end
%------------------------






%------------------------
% get MPC reaction forces
if strcmp(tsolver,'penalty')
    
    % get penalty value
    penval=fem.Options.Solver.PenaltyStiffness;
    
    nmp=length(fem.Boundary.Constraint.MPC); 
        
    for i=1:nmp
        
        id=fem.Boundary.Constraint.MPC(i).Id;
        coeff=fem.Boundary.Constraint.MPC(i).Coefficient;
        val=fem.Boundary.Constraint.MPC(i).Value;
        
        u=fem.Sol.U(id);
        
        r=-( dot(u, coeff) - val) * penval; % "-" for reaction force
        
        % save
        fem.Boundary.Constraint.MPC(i).Reaction=r;
        
    end

elseif strcmp(tsolver,'lagrange')
    
    % read # of dofs
    nSpc=fem.Sol.nLSPC;
    nMpc=fem.Sol.nLMPC; 

    for i=1:nMpc
        fem.Boundary.Constraint.MPC(i).Reaction=fem.Sol.RLamda(nSpc+i);
    end
    
else
    
    error('FEMP (Reaction force processing): Constraint handling method not recognised!') 
end
%------------------------






%------------------------
% get Unilateral constraint reaction forces
if strcmp(tsolver,'penalty')
    
    % get penalty value
    penval=fem.Options.Solver.PenaltyStiffness;
    
    nmp=length(fem.Boundary.Constraint.ContactWset); 
        
    for i=1:nmp
        
        id=fem.Boundary.Constraint.ContactWset(i).Id;
        coeff=fem.Boundary.Constraint.ContactWset(i).Coefficient;
        val=fem.Boundary.Constraint.ContactWset(i).Value;
        
        u=fem.Sol.U(id);
        
        r=-( dot(u, coeff) - val) * penval; % "-" for reaction force
        
        % save
        fem.Boundary.Constraint.ContactWset(i).Reaction.R=r;
        
    end

elseif strcmp(tsolver,'lagrange')
    
    % read # of dofs
    nSpc=fem.Sol.nLSPC;
    nMpc=fem.Sol.nLMPC; 
    nCt=fem.Sol.nLCt;  
    
    for i=1:nCt
        fem.Boundary.Constraint.ContactWset(i).Reaction.R=fem.Sol.RLamda(nSpc+nMpc+i);
    end
    
else
    
    error('FEMP (Reaction force processing): Constraint handling method not recognised!') 
end
%------------------------

disp('Reaction forces added!')





%-----------------------
function C=getDiff(A, B)

n=length(A);

C=zeros(1,n);
for i=1:n
    C(i)=A(i)-B(i);
end





