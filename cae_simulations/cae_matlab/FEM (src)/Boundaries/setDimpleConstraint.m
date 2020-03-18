% Calculate MPC related to dimple pairs
   
%..
function fem=setDimpleConstraint(fem)

% # of dimple pairs
nct=length(fem.Boundary.DimplePair);

% check
if nct==0
    return
end

% # of already defined MPCs for contact
nMpc=length(fem.Boundary.Constraint.ContactMPC);

disp('Calculating Dimple Pairs...')

% loop over dimple pairs
for kct=1:nct
                
        % read id of master and slave part
        P0=fem.Boundary.DimplePair(kct).Pm;
        idmaster=fem.Boundary.DimplePair(kct).Master;
        mastflip=fem.Boundary.DimplePair(kct).MasterFlip;

        idslave=fem.Boundary.DimplePair(kct).Slave;
        
        dsearch=fem.Boundary.DimplePair(kct).SearchDist;
        
        dimpleh=fem.Boundary.DimplePair(kct).Height;
        offseth=fem.Boundary.DimplePair(kct).Offset; % offset value
        
        frame=fem.Boundary.DimplePair(kct).Frame;

        %--
        %etype=fem.Boundary.DimplePair(kct).Physic;
        %--
        
        % flip normal for master component
        if mastflip
           fem=flipNormalComponent(fem,idmaster);
        end

        % SLAVE DOMAIN:
        flags=false;
        nodeslave=fem.Domain(idslave).Node;

         if ~isempty(nodeslave)

                % global search
                if strcmp(frame,'ref')
                    temp=fem.xMesh.Node.Coordinate(nodeslave,:);
                elseif strcmp(frame,'def')
                    temp=fem.Sol.DeformedFrame.Node.Coordinate(nodeslave,:);
                end
                    
                temp(:,1)=temp(:,1)-P0(1);
                temp(:,2)=temp(:,2)-P0(2);
                temp(:,3)=temp(:,3)-P0(3);

                dik=sqrt(sum(temp.^2,2));

                [dm, mid]=min(dik);
                 mid=nodeslave(mid);

                 % local search
                 mstslv=1; % slave definition
                 [flags,...
                   dofss,...
                   coeffus,...
                   coeffvs,...
                   coeffws,...
                   Pps,...
                   ~]=getProjection(fem, mid, P0, dsearch, mstslv, frame);

             if ~flags % look for nearest node (node-projection)

                    if fem.Options.UseActiveSelection % use selection
                        flagactive=fem.Selection.Node.Status(mid);
                    else
                        flagactive=true; % use any element
                    end

                    if dm<=dsearch && flagactive

                        dofss=fem.xMesh.Node.NodeIndex(mid,:);

                        % get only translations
                        dofss=dofss(:,1:3);
                        
                        % get coefficients
                        coeffus=1;
                        coeffvs=1;
                        coeffws=1;

                        Pps=fem.xMesh.Node.Coordinate(mid,:);

                        flags=true;

                    end
             end

         end


        % MASTER DOMAIN:
        flagm=false;
        nodemaster=fem.Domain(idmaster).Node;

         if ~isempty(nodemaster) && flags

                % global search
                if strcmp(frame,'ref')
                    temp=fem.xMesh.Node.Coordinate(nodemaster,:);
                elseif strcmp(frame,'def')
                    temp=fem.Sol.DeformedFrame.Node.Coordinate(nodemaster,:);
                end
                
                temp(:,1)=temp(:,1)-Pps(1);
                temp(:,2)=temp(:,2)-Pps(2);
                temp(:,3)=temp(:,3)-Pps(3);

                dik=sqrt(sum(temp.^2,2));

                [dm, mid]=min(dik);
                mid=nodemaster(mid);

                 % local search
                 mstslv=2; % master definition
                 [flagm,...
                       dofsm,...
                       coeffum,...
                       coeffvm,...
                       coeffwm,...
                       Ppm,...
                       Nm]=getProjection(fem, mid, Pps, dsearch, mstslv, frame);

             if ~flagm % look for the nearest node (node-projection)

                    if fem.Options.UseActiveSelection % use selection
                        flagactive=fem.Selection.Node.Status(mid);
                    else
                        flagactive=true; % use any element
                    end

                    if dm<=dsearch && flagactive

                        dofsm=fem.xMesh.Node.NodeIndex(mid,:);

                        % get only translations
                        dofsm=dofsm(:,1:3);

                        Ppm=fem.xMesh.Node.Coordinate(mid,:);

                        % get node normal vector
                        Nm=fem.xMesh.Node.Normal(mid,:);
                        
                        % get coefficients
                        coeffum=Nm(1);
                        coeffvm=Nm(2);
                        coeffwm=Nm(3);

                        flagm=true;

                    end
             end

         end


        % save data
        if flags && flagm

          % dofs
          dofss=dofss(:);
          dofsm=dofsm(:);

          dofs=[dofss
                dofsm];

          % update slave coefficient
          coeffus=coeffus*Nm(1);
          coeffvs=coeffvs*Nm(2);
          coeffws=coeffws*Nm(3);

          coeff=[coeffus
                 coeffvs
                 coeffws
                 coeffum
                 coeffvm
                 coeffwm];
             
          % save gap
          gsign=dot( (Pps-Ppm), Nm);

          %- add MPC constraints
          nMpc=nMpc+1;
          
          % save reference for reaction recovery
          fem.Boundary.DimplePair(kct).Reaction.Id=nMpc;
          fem.Boundary.DimplePair(kct).Reaction.Type='unilateral';

          fem.Boundary.Constraint.ContactMPC(nMpc).Value = -(gsign - dimpleh - offseth);

          % save ID
          fem.Boundary.Constraint.ContactMPC(nMpc).Id=dofs;

          % save coefficient
          fem.Boundary.Constraint.ContactMPC(nMpc).Coefficient=coeff;

          % save signed distance
          fem.Boundary.Constraint.ContactMPC(nMpc).Gap = gsign - dimpleh - offseth;  

          % save node coordinates
          fem.Boundary.Constraint.ContactMPC(nMpc).Psl=Pps; % slave
          fem.Boundary.Constraint.ContactMPC(nMpc).Pms=Ppm; % master

          fem.Boundary.Constraint.ContactMPC(nMpc).Type='assigned'; 

          fem.Boundary.Constraint.ContactMPC(nMpc).IdContactPair=-3; % -3 for dimple

          % 
          fem.Boundary.DimplePair(kct).Pms=Ppm;
          fem.Boundary.DimplePair(kct).Psl=Pps;

          fem.Boundary.DimplePair(kct).Type='assigned';


        else

           fem.Boundary.DimplePair(kct).Type='not-assigned';

        end
        
        % reset normal vectors for master component
        if mastflip
           fem=resetNormalComponent(fem,idmaster);
        end
         
end


disp('Dimple Pairs Added!')



% get projection
function [flag,...
               dofs,...
               coeffu,...
               coeffv,...
               coeffw,...
               Pp,...
               Nm]=getProjection(fem, mid, Ps, dsearch, mstslv, frame)


% mid: id of the closest node on the projected component
% Ps: projecting node coordinate
% dsearch: searching distance
% mstslv= 1/2 for slave or master component
% frame: frame reference (ref, def)

% set output
flag=false;
dofs=[];
coeffu=[];
coeffv=[];
coeffw=[];
Pp=[];
Nm=[]; % element normal
    
% num. tollerance allowed
eps=fem.Options.Eps;

% WORKING ON MASTER ELEMENT
idele=getMasterElement(fem, mid);

% loop over all elements
nele=length(idele);

for i=1:nele

    cetype=fem.xMesh.Element(idele(i)).Type;
    
      if fem.Options.UseActiveSelection % use selection
          flagactive=fem.Selection.Element.Status(idele(i));
      else
          flagactive=true; % use any element
      end
          
      if flagactive
        
            % only for quad and tria elements
            if strcmp(cetype,'quad') || strcmp(cetype,'tria')

                idm=fem.xMesh.Element(idele(i)).Element;

                if strcmp(frame,'ref')
                    
                    % get Point
                    Pe=fem.xMesh.Element(idele(i)).Tmatrix.P0lGeom;

                    % get Rotation matrix
                    R0l=fem.xMesh.Element(idele(i)).Tmatrix.T0lGeom;

                    % get points of element
                    Pmb=fem.xMesh.Node.Coordinate(idm,:);
                    
                elseif strcmp(frame,'def')
                    
                    % get Point
                    Pe=fem.Sol.DeformedFrame.Element(idele(i)).Tmatrix.P0lGeom;

                    % get Rotation matrix
                    R0l=fem.Sol.DeformedFrame.Element(idele(i)).Tmatrix.T0lGeom;

                    % get points of element
                    Pmb=fem.Sol.DeformedFrame.Node.Coordinate(idm,:);
                    
                end

                % check if "Ps" is inside the element
                Pmb=applyinv4x4(Pmb, R0l, Pe);

                % project "Ps" into element plane    
                Pp=applyinv4x4(Ps, R0l, Pe);

                inpol=pinPoly(Pp,Pmb,eps);

                % the point is inside the element
                if inpol

                  % use "z" coordinate as initial guess of the distance
                  mdist=abs(Pp(3));

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
if flag % use element projection
        
      % get dofs ids
      dofs=fem.xMesh.Element(elem).ElementNodeIndex(:,1:3);
      
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
     
      if strcmp(frame,'ref')
          
          % ... and normal vector
          Nm=fem.xMesh.Element(elem).Tmatrix.Normal;  

          % get projection point on the warped surface 
          Pe=fem.xMesh.Element(elem).Tmatrix.P0lGeom;
          R0l=fem.xMesh.Element(elem).Tmatrix.T0lGeom;

          % get node coordinates of the element
          Pmb0=fem.xMesh.Node.Coordinate(elemidm,:);
      
      elseif strcmp(frame,'def')
          
          % ... and normal vector
          Nm=fem.Sol.DeformedFrame.Element(elem).Tmatrix.Normal;  

          % get projection point on the warped surface 
          Pe=fem.Sol.DeformedFrame.Element(elem).Tmatrix.P0lGeom;
          R0l=fem.Sol.DeformedFrame.Element(elem).Tmatrix.T0lGeom;

          % get node coordinates of the element
          Pmb0=fem.Sol.DeformedFrame.Node.Coordinate(elemidm,:);
           
      end

      % save coefficient
      if mstslv==1 % slave
        coeffu=N';
        coeffv=N';
        coeffw=N';
      else % master
        coeffu=-Nm(1)*N';
        coeffv=-Nm(2)*N';
        coeffw=-Nm(3)*N';
      end

      
      Pmb=applyinv4x4(Pmb0, R0l, Pe);
           
      Pp=N*Pmb;
      
      Pp=apply4x4(Pp, R0l, Pe);
                  
end

