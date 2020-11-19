% define nodal forces from element-based forces

% N.B.: NOMINAL GEOMETRY IS GIVEN IN INPUT
    
%..
function  fem=setLoadElement(fem)

% read number of load
nc=length(fem.Boundary.Load.Element);

%-
if nc==0 
    return
end

% loop
for i=1:nc

    % read inputs        
    ref=fem.Boundary.Load.Element(i).Reference; % reference
        
    P0=fem.Boundary.Load.Element(i).Pm;
    values=fem.Boundary.Load.Element(i).Value;
    iddom=fem.Boundary.Load.Element(i).Domain;
     
    %--
    %etype=fem.Boundary.Constraint.Bilateral.Element(i).Physic;
    %--
     
    % search distance
    dsearch=fem.Boundary.Load.Element(i).SearchDist;
     
    % initialise
    fem.Boundary.Load.Element(i).Type='not-assigned';
          
    % start search                                                        
    flag=false;
    
    nodedom=fem.Domain(iddom).Node;

    if ~isempty(nodedom) 

        %----------------------
        % STEP 1: GLOBAL SEARCH
        temp=fem.xMesh.Node.Coordinate(nodedom,:); tempd=temp;
        tempd(:,1)=temp(:,1)-P0(1);
        tempd(:,2)=temp(:,2)-P0(2);
        tempd(:,3)=temp(:,3)-P0(3);

        dik=sqrt(sum(tempd.^2,2));
        [~, midt]=min(dik); % dm
        mid=nodedom(midt);
        
        %----------------------
        % STEP 2: LOCAL SEARCH
        [flag,...
            dofsele,...
            coeff,...
            Pp]=getProjection(fem,mid,P0,dsearch);
        
        if ~flag % check node to node
             Pp=temp(midt,:);
             mdist=norm(P0-Pp);
             if mdist<=dsearch

                  % get coefficients
                  coeff=1;

                  % get dofs
                  dofsele=fem.xMesh.Node.NodeIndex(mid,:);

                  flag=true;
             end 
         end

    end
        
    % save all for element
    if flag
            
        % save projection point
        fem.Boundary.Load.Element(i).Pms=Pp;

        fem.Boundary.Load.Element(i).Type='element';
            
        %---------------------------------------------------
        if strcmp(ref,'cartesian')
            
            % dofs
            dofs=fem.Boundary.Load.Element(i).DoF;

            ndofs=length(dofs);

            % get only those degrees of freedom related to "dofs"
            for jd=1:ndofs
                 fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId, dofsele(:,dofs(jd))'];
                 fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values(jd) * coeff];
            end
        
        %---------------------------------------------------
        elseif strcmp(ref,'vectorTra')
            
            vector=fem.Boundary.Load.Element(i).Nm;

            % save ID
            dofselek=dofsele(:,1:3);

            % save Coefficient
            coeffk=[coeff*vector(1),coeff*vector(2),coeff*vector(3)];

            fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,dofselek(:)'];
            fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values*coeffk];
                  
        %---------------------------------------------------
        elseif strcmp(ref,'vectorRot')
            
            vector=fem.Boundary.Load.Element(i).Nm;

            % save ID
            dofselek=dofsele(:,4:6);

            % save Coefficient
            coeffk=[coeff*vector(1),coeff*vector(2),coeff*vector(3)];

            fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,dofselek(:)'];
            fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values*coeffk];
            
        end
        
                %     else
                %         
                %         % save for closest node
                %         if fem.Options.UseActiveSelection % use selection
                %            flagactive=fem.Selection.Node.Status(mid);
                %         else
                %            flagactive=true; % use any nodes
                %         end
                %                         
                %         if dm<=dsearch && flagactive
                %                           
                %             %----
                %             Pp=fem.xMesh.Node.Coordinate(mid,:);
                %             fem.Boundary.Load.Element(i).Pms=Pp;
                % 
                %             fem.Boundary.Load.Element(i).Type='node';
                %             
                %             %---
                %             DoFs=fem.xMesh.Node.NodeIndex(mid,:);
                %               
                %             if strcmp(ref,'cartesian')
                %                      
                %                 dofs=fem.Boundary.Load.Element(i).DoF;
                %                 dofs=DoFs(dofs);  
                % 
                %                 % save all
                %                 fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,dofs];
                %                 fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values];
                %                                         
                %             %--------------------
                %             elseif strcmp(ref,'vectorTra')
                %                 
                %                   Nm=fem.Boundary.Load.Element(i).Nm; % unit vector
                % 
                %                   % get only traslations
                %                   dofs=DoFs(1:3);
                %                   
                %                   for jd=1:3
                %                       fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,dofs(jd)];
                %                       fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values*Nm(jd)];
                %                   end
                % 
                %           
                %             %--------------------
                %             elseif strcmp(ref,'vectorRot')
                %                 
                %                   Nm=fem.Boundary.Load.Element(i).Nm; % unit vector
                % 
                %                   % get only rotations
                %                   dofs=DoFs(4:6);
                %                   
                %                   for jd=1:3
                %                       fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,dofs(jd)];
                %                       fem.Boundary.Load.Value=[fem.Boundary.Load.Value, values*Nm(jd)];
                %                   end
                %                   
                %             end
                %             
                %             
                %         end
                %         
                %         
    end
       
end


%-------------------------------------------------------------------------
% get distance from master elements
function [flag,...
               dofs,...
               coeff,...
               Pp]=getProjection(fem,mid,Ps,dsearch)

% set output
flag=false;
dofs=[];
coeff=[];
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

                % get Point
                Pe=fem.xMesh.Element(idele(i)).Tmatrix.P0lGeom;

                % get Rotation matrix
                R0l=fem.xMesh.Element(idele(i)).Tmatrix.T0lGeom;

                % get points of element
                Pmb=fem.xMesh.Node.Coordinate(idm,:);

                % check if "Ps" is inside the element
                Pmb=applyinv4x4(Pmb, R0l, Pe);

                % project "Pm" into element plane    
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
if flag 
        
      % get dofs ids
      dofs=fem.xMesh.Element(elem).ElementNodeIndex;
      
      % get coefficient
      if strcmp(fem.xMesh.Element(elem).Type,'quad')
         [csips,etaps]=mapxy2csietaQuad4(Pxy(1),Pxy(2),Pmb);

         % ... and related weight
         [coeff, ~]=getNdNquad4node(csips,etaps);
      elseif strcmp(fem.xMesh.Element(elem).Type,'tria')
         [csips,etaps]=mapxy2csietaTria3(Pxy(1),Pxy(2),Pmb);

         % ... and related weight
         [coeff, ~]=getNdNtria3node(csips,etaps);

         %.........................


      end
     
      % get projection point on the warped surface 
      Pe=fem.xMesh.Element(elem).Tmatrix.P0lGeom;
      R0l=fem.xMesh.Element(elem).Tmatrix.T0lGeom;
      
      % get node coordinates of the element
      Pmb0=fem.xMesh.Node.Coordinate(elemidm,:);
      
      Pmb=applyinv4x4(Pmb0, R0l, Pe);
           
      Pp=coeff*Pmb;
      
      Pp=apply4x4(Pp, R0l, Pe);
            
end
          





