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

            % MASTER COMPONENT
            flag=false;
            if ~isempty(nodedom) 

                    %----------------------
                    % STEP 1: GLOBAL SEARCH
                    if strcmp(frame,'ref')
                        temp=fem.xMesh.Node.Coordinate(nodedom,:);
                    elseif strcmp(frame,'def')
                        temp=fem.Sol.DeformedFrame.Node.Coordinate(nodedom,:);
                    end
                    
                    temp(:,1)=temp(:,1)-P0(1);
                    temp(:,2)=temp(:,2)-P0(2);
                    temp(:,3)=temp(:,3)-P0(3);
                    
                    dik=sqrt(sum(temp.^2,2));

                    [~, mid]=min(dik);
                    mid=nodedom(mid);
                    
                    %----------------------
                    % STEP 2: LOCAL SEARCH
                    [flag,...
                          dofsele,...
                          coefu,...
                          coefv,...
                          coefw,...
                          gsign,...
                          Pp]=getProjection(fem, mid, P0, N0, dsearch, frame);
                                          
            end    
                    
            % calculate related Dofs for idk ele
            if flag
                  
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


% get projection point and its dofs
function [flag,...
          dofs,...
          coefu,...
          coefv,...
          coefw,...
          gsign,...
          Pp]=getProjection(fem, mid, P0, N0, dsearch, frame)

% flag: true/false => proejction found or not
% dofs: dofs involved in the constraint: [u, v, w]
% coef: coefficient to be included into MPC formulation
% Pp: projected master node

% mid: master id node from which I move around
% P0/N0: point and vector
% dsearch: search distance
% frame used for calculating the projected point

% set initial value:
flag=false;

dofs=[]; % 3/4 x 3... first column has the "u"; second column has the "v"; ... "w"

coefu=[]; 
coefv=[];
coefw=[];

gsign=[];

Pp=[];

% num. tollerance allowed
eps=fem.Options.Eps;

% WORKING ON MASTER ELEMENT
idele=getMasterElement(fem, mid);

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

                Pp0=P0+t*N0;

                % check is this point belongs to the boundary of the element

                % transform nodes
                Pmb=applyinv4x4(Pmb0, R0l, Pe);

                % transform projection point
                Pp=applyinv4x4(Pp0, R0l, Pe);

                % point inside polygon
                inpol=pinPoly(Pp,Pmb,eps);   

                % the point is inside the element
                if inpol

                  % projected point - key point
                  mdist=norm(Pp0-P0);

                  if mdist<=dsearch

                       % element id
                       elem=idele(i);

                       elemidm=idm;

                       % projected slave node on the element plane
                       Pxy=Pp(1:2);

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
      dofs=fem.xMesh.Element(elem).ElementNodeIndex(:,1:3); % only translation
      
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
      coefu=N0(1)*N';
      coefv=N0(2)*N';
      coefw=N0(3)*N';
          
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
                
      % signed distance
      gsign=dot( (Pp-P0), N0 );
      
end

    



