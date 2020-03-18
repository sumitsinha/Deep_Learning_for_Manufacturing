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
            
            % MASTER COMPONENT
            flagm=false;
            if ~isempty(nodemaster) 

                    %----------------------
                    % STEP 1: GLOBAL SEARCH
                    if strcmp(frame,'ref')
                        temp=fem.xMesh.Node.Coordinate(nodemaster,:);
                    elseif strcmp(frame,'def')
                        temp=fem.Sol.DeformedFrame.Node.Coordinate(nodemaster,:);
                    end
                    
                    temp(:,1)=temp(:,1)-P0(1);
                    temp(:,2)=temp(:,2)-P0(2);
                    temp(:,3)=temp(:,3)-P0(3);
                    
                    dik=sqrt(sum(temp.^2,2));

                    [~, mid]=min(dik);
                    mid=nodemaster(mid);
                    
                    if fem.Options.UseActiveSelection % use selection
                          flagactive=fem.Selection.Node.Status(mid);
                    else
                          flagactive=true; % use any node
                    end

                    if flagactive
                    
                        %----------------------
                        % STEP 2: LOCAL SEARCH

                        % find parameters for MPC
                        [flagm,...
                         dofsm,...
                         coefm,...
                         Pm]=getProjection(fem, mid, P0, N0, dsearch, frame);
                 
                    end

            end
            
            % SLAVE COMPONENT
            flags=false;
            if ~isempty(nodeslave) 

                    %----------------------
                    % STEP 1: GLOBAL SEARCH
                    if strcmp(frame,'ref')
                        temp=fem.xMesh.Node.Coordinate(nodeslave,:);
                    elseif strcmp(frame,'def')
                        temp=fem.Sol.DeformedFrame.Node.Coordinate(nodeslave,:);
                    end
                    
                    temp(:,1)=temp(:,1)-P0(1);
                    temp(:,2)=temp(:,2)-P0(2);
                    temp(:,3)=temp(:,3)-P0(3);
                    
                    dik=sqrt(sum(temp.^2,2));

                    [~, mid]=min(dik);
                    mid=nodeslave(mid);
                    
                    if fem.Options.UseActiveSelection % use selection
                          flagactive=fem.Selection.Node.Status(mid);
                    else
                          flagactive=true; % use any node
                    end

                    if flagactive
                    
                        %----------------------
                        % STEP 2: LOCAL SEARCH

                        % find parameters for MPC
                        [flags,...
                         dofss,...
                         coefs,...
                         Ps]=getProjection(fem, mid, P0, N0, dsearch, frame);
                 
                    end

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


% get projection point and its dofs
function [flag,...
          dofs,...
          coef,...
          Pp]=getProjection(fem, mid, P0, N0, dsearch, frame)

% flag: true/false => proejction found or not
% dofs: dofs involved in the constraint: [u, v, w]
% coef: coefficient to be included into MPC formulation
% Pp: projected master node

% mid: master id node from which I move around
% P0/N0: point and vector
% dsearch: search distance
% frame: frame definiton (ref, def)

% set initial value:
flag=false;

dofs=[]; % 3/4 x 3... first column has the "u"; second column has the "v"; ... "w"

coef=[]; 

Pp=[];

% num. tolerance allowed
eps=fem.Options.Eps;

% WORKING ON MASTER ELEMENT
idele=getMasterElement(fem, mid);

% loop over all elements
nele=length(idele);

for i=1:nele
    
    etype=fem.xMesh.Element(idele(i)).Type;
    
       if fem.Options.UseActiveSelection % use selection
              flagactive=fem.Selection.Element.Status(idele(i));
       else
              flagactive=true; % use any element
       end
          
       if flagactive
        
            % only for quad and tria elements
            if strcmp(etype,'quad') || strcmp(etype,'tria')

                idm=fem.xMesh.Element(idele(i)).Element;

                
                if strcmp(frame,'ref')
                    
                    % get node coordinates of the element
                    Pmb0=fem.xMesh.Node.Coordinate(idm,:);

                    % get Rotation matrix
                    R0l=fem.xMesh.Element(idele(i)).Tmatrix.T0lGeom;

                    % get origin of local coordinate frame
                    Pe=fem.xMesh.Element(idele(i)).Tmatrix.P0lGeom;

                    % normal vector to the element
                    Ne=fem.xMesh.Element(idele(i)).Tmatrix.Normal;
                
                elseif strcmp(frame,'def')
                    
                    % get node coordinates of the element
                    Pmb0=fem.Sol.DeformedFrame.Node.Coordinate(idm,:);

                    % get Rotation matrix
                    R0l=fem.Sol.DeformedFrame.Element(idele(i)).Tmatrix.T0lGeom;

                    % get origin of local coordinate frame
                    Pe=fem.Sol.DeformedFrame.Element(idele(i)).Tmatrix.P0lGeom;

                    % normal vector to the element
                    Ne=fem.Sol.DeformedFrame.Element(idele(i)).Tmatrix.Normal;
                    
                end

                % find intersection
                t=dot( (Pe-P0), Ne ) / dot ( N0, Ne );

                Pp=P0+t*N0;

                % check is this point belongs to the boundary of the element

                % transform nodes
                Pmb=applyinv4x4(Pmb0, R0l, Pe);

                % transform projection point
                Pp=applyinv4x4(Pp, R0l, Pe);

                % point inside polygon
                inpol=pinPoly(Pp,Pmb,eps);   

                % the point is inside the element
                if inpol

                  % use "z" coordinate as initial guess of the distance
                  mdist=abs(Pp(3));

                  if mdist<=dsearch
                     Pxy=Pp(1:2);


                       % element id
                       elem=idele(i);

                       elemidm=idm;

                       % projected slave node on the element plane
                     flag=true;

                       break

                  end

                end
                
            end
            
       end
    
end % next element


% at least one projection found
if flag    
    
      % get dofs ids
      dofs=fem.xMesh.Element(elem).ElementNodeIndex; 
      
      % get coefficient
      if strcmp(fem.xMesh.Element(elem).Type,'quad')
         [csips,etaps]=mapxy2csietaQuad4(Pxy(1),Pxy(2),Pmb);

         % ... and related weight
         [N, ~]=getNdNquad4node(csips,etaps);
      elseif strcmp(fem.xMesh.Element(elem).Type,'tria')
         [csips,etaps]=mapxy2csietaTria3(Pxy(1),Pxy(2),Pmb);

         % ... and related weight
         [N, ~]=getNdNtria3node(csips,etaps);

         %.........................

      end
     
      %--     
      coef=N';
      
      % get projection point on the warped surface
      if strcmp(frame,'ref')
          
          Pe=fem.xMesh.Element(elem).Tmatrix.P0lGeom;
          R0l=fem.xMesh.Element(elem).Tmatrix.T0lGeom;

          % get node coordinates of the element
          Pmb0=fem.xMesh.Node.Coordinate(elemidm,:);
          
      elseif strcmp(frame,'def')
          
          Pe=fem.Sol.DeformedFrame.Element(elem).Tmatrix.P0lGeom;
          R0l=fem.Sol.DeformedFrame.Element(elem).Tmatrix.T0lGeom;

          % get node coordinates of the element
          Pmb0=fem.Sol.DeformedFrame.Node.Coordinate(elemidm,:);
          
      end
      
      
      Pmb=applyinv4x4(Pmb0, R0l, Pe);
           
      Pp=N*Pmb;
      
      Pp=apply4x4(Pp, R0l, Pe);
      
      % project the point on the N0 direction
      t=dot( (Pp-P0), N0 );
      Pp=P0+t*N0;
           
end
          

