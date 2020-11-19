% Calculate MPC related to rigid links
   
%..
function fem=setRigidLinkConstraint(fem)

% # of links
nct=length(fem.Boundary.Constraint.RigidLink);

% check
if nct==0
    return
end

% # of already defined MPCs
nMpc=length(fem.Boundary.Constraint.MPC);

% solver type:
tsolver=fem.Options.Solver.Method;

disp('Calculating Rigid Links...')

% loop over all links            
for kct=1:nct

            % read id of master and slave part
            idmaster=fem.Boundary.Constraint.RigidLink(kct).Master;
            idslave=fem.Boundary.Constraint.RigidLink(kct).Slave;
            dsearch=fem.Boundary.Constraint.RigidLink(kct).SearchDist;
            
            P0=fem.Boundary.Constraint.RigidLink(kct).Pm;
            N0=fem.Boundary.Constraint.RigidLink(kct).Nm;
            
            frame=fem.Boundary.Constraint.RigidLink(kct).Frame;
     
            % init.
            fem.Boundary.Constraint.RigidLink(kct).Type='not-assigned';
             
            nodemaster=fem.Domain(idmaster).Node;
            nodeslave=fem.Domain(idslave).Node;
            
            % SLAVE COMPONENT
            flags=false;
            if ~isempty(nodeslave) 
                    [flags,...
                      dofss,...
                      coefu,...
                      coefv,...
                      coefw,...
                      ~,...
                      Ps]=getProjectionMPC(fem, P0, N0, idslave, dsearch, frame); 
                      coefs=[coefu, coefv, coefw];

            end
            
            % MASTER COMPONENT
            flagm=false;
            if ~isempty(nodemaster) 
                [flagm,...
                  dofsm,...
                  coefu,...
                  coefv,...
                  coefw,...
                  ~,...
                  Pm]=getProjectionMPC(fem, P0, N0, idmaster, dsearch, frame);   
                  coefm=[coefu, coefv, coefw];

            end
                        
            %--
            if flags && flagm
                
                % save reference for reaction recovery
                fem.Boundary.Constraint.RigidLink(kct).Reaction.Type='mpc';
                fem.Boundary.Constraint.RigidLink(kct).Reaction.Id=[];

                %----------------------------
                for j=1:6 % define constraint along X, Y and Z directions and relative rotations
                    
                      % dofs
                      dofs=[dofss(:,j)
                              dofsm(:,j)];

                      % coefficient
                      coeff=[coefs
                               -coefm];

                      %--------------------
                      gsign=0;
                      %--------------------

                      %- add MPC constraints
                      nMpc=nMpc+1;
                      
                      % save reference for reaction recovery
                      fem.Boundary.Constraint.RigidLink(kct).Reaction.Id=[fem.Boundary.Constraint.RigidLink(kct).Reaction.Id nMpc];

                      fem.Boundary.Constraint.MPC(nMpc).Id=dofs;

                      % save Coefficient
                      fem.Boundary.Constraint.MPC(nMpc).Coefficient=coeff;

                      fem.Boundary.Constraint.MPC(nMpc).Value=gsign;

                      % save projection points
                      fem.Boundary.Constraint.RigidLink(kct).Pms=Pm;
                      fem.Boundary.Constraint.RigidLink(kct).Psl=Ps;
                      fem.Boundary.Constraint.RigidLink(kct).Type='assigned';

                      % update # of non zero-entries
                     if strcmp(tsolver,'lagrange')
                         fem.Sol.Kast.n=fem.Sol.Kast.n+length(coeff)*2;
                     elseif strcmp(tsolver,'penalty')
                         fem.Sol.Kast.n=fem.Sol.Kast.n+length(coeff)^2;
                     else

                         error('FEMP (Refreshing): Constraint handling method not recognised!') 
                     end
                 
                end
                                    
            end
end

disp('Rigid Links Added!')

