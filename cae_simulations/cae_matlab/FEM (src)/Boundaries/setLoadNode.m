% set node loads

%..
function fem=setLoadNode(fem)

% read number of constraints
nc=length(fem.Boundary.Load.Node);

%-
if nc==0 
    return
end

for i=1:nc
    
    idnode=fem.Boundary.Load.Node(i).Node; %% id node
    idof=fem.Boundary.Load.Node(i).DoF; % id dofs
    ival=fem.Boundary.Load.Node(i).Value; % load value
    
    ref=fem.Boundary.Load.Node(i).Reference; 
    
    %--
    %etype=fem.Boundary.Load.Node(i).Physic;
    %--
   
        for j=1:length(idnode)
            
            fem.Boundary.Load.Node(i).Type{j}='not-assigned';

            if fem.Options.UseActiveSelection % use selection
              flagactive=fem.Selection.Node.Status(idnode(j));
            else
              flagactive=true; % use any element
            end
          
            if flagactive

                fem.Boundary.Load.Node(i).Type{j}='assigned';
                
                if strcmp(ref,'cartesian')
                    
                     % dofs...
                     DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);

                     % get only those degrees of freedom related to "dofs"          
                     fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,DoFs(idof)];

                     fem.Boundary.Load.Value=[fem.Boundary.Load.Value,ival];
                 
                elseif strcmp(ref,'vectorTra')
                    
                     % vector...
                     vector=fem.Boundary.Load.Node(i).Nm;

                     % dofs
                     DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);
                     
                     fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,DoFs(1:3)]; % only translation

                     fem.Boundary.Load.Value=[fem.Boundary.Load.Value,ival*vector];
        
                elseif strcmp(ref,'vectorRot')
                    
                     % vector...
                     vector=fem.Boundary.Load.Node(i).Nm;

                     % dofs
                     DoFs=fem.xMesh.Node.NodeIndex(idnode(j),:);
                     
                     fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId,DoFs(4:6)]; % only rotations

                     fem.Boundary.Load.Value=[fem.Boundary.Load.Value,ival*vector];
        
                end
    
             
            end

        end % no. of node loop
        
end % no. of constraint loop


