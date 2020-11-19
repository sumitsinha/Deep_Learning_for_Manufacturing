% Calculate Constraints related to unilateral constraints

%..
function fem=setUnilateralConstraint(fem)

% read number of constraints
nc=length(fem.Boundary.Constraint.Unilateral);

%-
if nc==0 
    return
end

% solver type:
tsolver=fem.Options.Solver.Method;

% loop
for i=1:nc

     % read inputs   
     Pm=fem.Boundary.Constraint.Unilateral(i).Pm;
     N0=fem.Boundary.Constraint.Unilateral(i).Nm;
     
     sizeFlag=fem.Boundary.Constraint.Unilateral(i).Size;
     
     % constraint size enabled
     if sizeFlag
         
         if ~isempty(fem.Boundary.Constraint.Unilateral(i).Pmsize)
             
             P=fem.Boundary.Constraint.Unilateral(i).Pmsize;
             
         else
             
             Nt=fem.Boundary.Constraint.Unilateral(i).Nt;
             Ac=fem.Boundary.Constraint.Unilateral(i).SizeA;
             Bc=fem.Boundary.Constraint.Unilateral(i).SizeB;
             
             % get additional point for constraint "i-th"
             P=getSize2Corner(N0, Pm, sizeFlag, Nt, Ac, Bc);
         
         end
     else
         
         P=Pm;
         
     end
     
     iddom=fem.Boundary.Constraint.Unilateral(i).Domain;
     
     offgap=fem.Boundary.Constraint.Unilateral(i).Offset;
     
     % search distance
     dsearch=fem.Boundary.Constraint.Unilateral(i).SearchDist;
     
     % get frame
     frame=fem.Boundary.Constraint.Unilateral(i).Frame;
     
     % set initial field for reaction forces
     fem.Boundary.Constraint.Unilateral(i).Reaction.Id=[];
     
     % save out
     fem.Boundary.Constraint.Unilateral(i).Pmsize=P;
          
     nodedom=fem.Domain(iddom).Node;

     for ip=1:size(P,1) % loop over constraint points

            % init
            fem.Boundary.Constraint.Unilateral(i).Type{ip}='not-assigned';
     
            % get ip-th point
            P0=P(ip,:);

            % Projection
            flag=false;
            if ~isempty(nodedom) 
                [flag,...
                  dofsele,...
                  coefu,...
                  coefv,...
                  coefw,...
                  gsign,...
                  Pp]=getProjectionMPC(fem, P0, N0, iddom, dsearch, frame);                         
            end    
                    
            % save back
            if flag
                  dofsele=dofsele(:,1:3); % Take only translations
                  dofsele=dofsele(:);
                  
                  coeff=[coefu
                         coefv
                         coefw];
                  
                  % save...
                  optcon=fem.Boundary.Constraint.Unilateral(i).Constraint;
                  
                  if strcmp(optcon,'free') % unilateral
                     
                      % add MPC
                      nMpc=length(fem.Boundary.Constraint.ContactMPC)+1;
                      
                      % save reference for reaction recovery
                      fem.Boundary.Constraint.Unilateral(i).Reaction.Type='unilateral';
                      fem.Boundary.Constraint.Unilateral(i).Reaction.Id=[fem.Boundary.Constraint.Unilateral(i).Reaction.Id nMpc];

                      % save ID
                      fem.Boundary.Constraint.ContactMPC(nMpc).Id=dofsele;

                      % save coefficient
                      fem.Boundary.Constraint.ContactMPC(nMpc).Coefficient=coeff;

                      % save value
                      fem.Boundary.Constraint.ContactMPC(nMpc).Value=-(gsign-offgap); % offset added #OK

                      % save gap
                      fem.Boundary.Constraint.ContactMPC(nMpc).Gap=gsign-offgap;

                      fem.Boundary.Constraint.ContactMPC(nMpc).Type='unilateral';

                      fem.Boundary.Constraint.ContactMPC(nMpc).Pms=Pp;
                      fem.Boundary.Constraint.ContactMPC(nMpc).Psl=Pp;
                      fem.Boundary.Constraint.ContactMPC(nMpc).IdContactPair=-1; % "-1" for unilateral pair
                      
                  elseif  strcmp(optcon,'lock') % lock option
                      
                      % add MPC
                      nMpc=length(fem.Boundary.Constraint.MPC)+1;

                      % save reference for reaction recovery
                      fem.Boundary.Constraint.Unilateral(i).Reaction.Type='mpc';
                      fem.Boundary.Constraint.Unilateral(i).Reaction.Id=[fem.Boundary.Constraint.Unilateral(i).Reaction.Id nMpc];
                      
                      % save ID
                      fem.Boundary.Constraint.MPC(nMpc).Id=dofsele;

                      % save Coefficient
                      fem.Boundary.Constraint.MPC(nMpc).Coefficient=coeff;

                      fem.Boundary.Constraint.MPC(nMpc).Value=-(gsign-offgap);
                      
                      % update # of non zero-entries
                      if strcmp(tsolver,'lagrange')
                          fem.Sol.Kast.n=fem.Sol.Kast.n+length(coeff)*2;
                      elseif strcmp(tsolver,'penalty')
                          fem.Sol.Kast.n=fem.Sol.Kast.n+length(coeff)^2;
                      else
    
                          error('FEMP (Refreshing): Constraint handling method not recognised!') 
                      end
          
                      
                  end
                      
                 %-
                 fem.Boundary.Constraint.Unilateral(i).Pms(ip,:)=Pp; 
                 fem.Boundary.Constraint.Unilateral(i).Type{ip}='assigned';
                      
             
            end
            
     end

end

