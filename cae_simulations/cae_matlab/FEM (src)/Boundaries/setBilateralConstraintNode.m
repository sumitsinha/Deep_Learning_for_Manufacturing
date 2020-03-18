% Calculate bilateral constraints

%..
function fem=setBilateralConstraintNode(fem)

% read number of constraints
fem.Sol.nLSPC=0;
nc=length(fem.Boundary.Constraint.Bilateral.Node);

%-
if nc==0 
    return
end

% # of MPC already defined
nMpc=length(fem.Boundary.Constraint.MPC);

% solver type:
tsolver=fem.Options.Solver.Method;


%--
fem.Boundary.Constraint.SPC.Id=[];
fem.Boundary.Constraint.SPC.Value=[]; 

for i=1:nc
    
    %--
    ref=fem.Boundary.Constraint.Bilateral.Node(i).Reference;
    
    %--
    idnode=fem.Boundary.Constraint.Bilateral.Node(i).Node; %% id node   
    
    %--
    ival=fem.Boundary.Constraint.Bilateral.Node(i).Value; % constraint value
    
    %--
    %%% etype=fem.Boundary.Constraint.Bilateral.Node(i).Physic;
    %--
    
    % cartesian constraint
    if strcmp(ref,'cartesian')
       
       % dofs
       idof=fem.Boundary.Constraint.Bilateral.Node(i).DoF; % id dofs
        
       for j=1:length(idnode)
          
          fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='not-assigned';

          % dofs...
          if fem.Options.UseActiveSelection % use selection
              flagactive=fem.Selection.Node.Status(idnode(j));
          else
              flagactive=true; % use any nodes
          end
          
          if flagactive
                   
                 DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);
                 
                 % save reference for reaction recovery
                 for jj=1:length(idof)     
                     
                     fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Type{j}{jj}='spc';
                     fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Id{j}(jj)=length(fem.Boundary.Constraint.SPC.Id)+1;
                     
                     fem.Boundary.Constraint.SPC.Id=[fem.Boundary.Constraint.SPC.Id,DoFs(idof(jj))];
                     fem.Boundary.Constraint.SPC.Value=[fem.Boundary.Constraint.SPC.Value,ival(jj)];
                     
                 end
       
                 % update # of non zero-entries
                 if strcmp(tsolver,'lagrange')
                    fem.Sol.Kast.n=fem.Sol.Kast.n+length(ival)*2; 
                 elseif strcmp(tsolver,'penalty')
                    fem.Sol.Kast.n=fem.Sol.Kast.n+length(ival); 
                 else

                     error('FEMP (Refreshing): Constraint handling method not recognised!') 
                 end

                 %--
                 fem.Sol.nLSPC=fem.Sol.nLSPC+length(ival);
                 
                 fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='assigned';
             
          end
             
        end % no. of node loop
             

    % translation along given direction
    elseif strcmp(ref,'vectorTra')
        
        %--
        Nm=fem.Boundary.Constraint.Bilateral.Node(i).Nm; % unit vector
    
        for j=1:length(idnode)
          
          fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='not-assigned';
            
          % dofs...
          if fem.Options.UseActiveSelection % use selection
              flagactive=fem.Selection.Node.Status(idnode(j));
          else
              flagactive=true; % use any nodes
          end
          
          if flagactive
              
                  DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);

                  % get only traslations
                  DoFs=DoFs(1:3);

                  % add MPC
                  nMpc=nMpc+1;

                  % save reference for reaction recovery
                  fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Type='mpc';
                  fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Id=nMpc;

                  % save ID
                  fem.Boundary.Constraint.MPC(nMpc).Id=DoFs;

                  % save Coefficient
                  fem.Boundary.Constraint.MPC(nMpc).Coefficient=Nm;

                  fem.Boundary.Constraint.MPC(nMpc).Value=ival;

                  % update # of non zero-entries
                  if strcmp(tsolver,'lagrange')
                       fem.Sol.Kast.n=fem.Sol.Kast.n+6;  % => 3*2
                  elseif strcmp(tsolver,'penalty')
                       fem.Sol.Kast.n=fem.Sol.Kast.n+9;  % => 3*3
                  else

                         error('FEMP (Refreshing): Constraint handling method not recognised!') 
                  end
                  
                  fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='assigned';


          end
          
        end
     
     % rotations around given direction
     elseif strcmp(ref,'vectorRot')
        
        %--
        Nm=fem.Boundary.Constraint.Bilateral.Node(i).Nm; % unit vector
        
        for j=1:length(idnode)
          
          fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='not-assigned';
          
          % dofs...
          if fem.Options.UseActiveSelection % use selection
              flagactive=fem.Selection.Node.Status(idnode(j));
          else
              flagactive=true; % use any nodes
          end
          
          if flagactive
              
                  DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);
          
                  % get only rotations
                  DoFs=DoFs(4:6);

                  % add MPC
                  nMpc=nMpc+1;

                  % save reference for reaction recovery
                  fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Type='mpc';
                  fem.Boundary.Constraint.Bilateral.Node(i).Reaction.Id=nMpc;

                  % save ID
                  fem.Boundary.Constraint.MPC(nMpc).Id=DoFs;

                  % save Coefficient
                  fem.Boundary.Constraint.MPC(nMpc).Coefficient=Nm;

                  fem.Boundary.Constraint.MPC(nMpc).Value=ival;

                  % update # of non zero-entries
                  if strcmp(tsolver,'lagrange')
                       fem.Sol.Kast.n=fem.Sol.Kast.n+6; % => 3*2
                  elseif strcmp(tsolver,'penalty')
                       fem.Sol.Kast.n=fem.Sol.Kast.n+9; % => 3*3
                  else

                         error('FEMP (Refreshing): Constraint handling method not recognised!') 
                  end
                  
                  fem.Boundary.Constraint.Bilateral.Node(i).Type{j}='assigned';
          
          end
          
        end
    end
             
end % no. of constraints



