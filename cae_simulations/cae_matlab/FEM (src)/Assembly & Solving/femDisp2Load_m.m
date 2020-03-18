% Re-built nodal forces based on nodal displacements
% N.B.: Initial stress due to given node displacements

% sigma=D*B*u
% force=-int(Bt*D*B*u) => force=-K*u

function fem=femDisp2Load(fem, U)

% check if the element is active
if fem.Options.UseActiveSelection % use selection
  
      % save counters
      ndofs=fem.Selection.Element.Tria.Count * 18^2 + fem.Selection.Element.Quad.Count * 24^2; 
      
      nele=fem.Selection.Element.Tria.Count+fem.Selection.Element.Quad.Count;
    
else
    
      nd=fem.Sol.nDom;

      ndofs=0;
      for i=1:nd
        ndofs = ndofs + length(fem.Domain(i).ElementTria) * 18^2 + length(fem.Domain(i).ElementQuad) * 24^2;
      end
      
      nele=length(fem.xMesh.Element);

end

% take a copy of previuos loads
tempid=fem.Boundary.Load.DofId;
tempvalue=fem.Boundary.Load.Value;

ntemp=length(tempid);

% update all
n=ndofs+ntemp;
fem.Boundary.Load.DofId=zeros(1,n);
fem.Boundary.Load.Value=zeros(1,n);

fem.Boundary.Load.DofId(1:ntemp)=tempid;
fem.Boundary.Load.Value(1:ntemp)=tempvalue;

% loop over all elements
count=ntemp+1;
for i=1:nele
    
     % check if the element is active
      if fem.Options.UseActiveSelection % use selection
          flagactive=fem.Selection.Element.Status(i);
      else
          flagactive=true; % use any element
      end

      if flagactive

         % get stiffness
         Ki=fem.xMesh.Element(i).Ke;
         
         % get dofs
         dofs=fem.xMesh.Element(i).ElementIndex;
         
         % get displacement field
         ui=U(dofs);
         
         % get forces
         Fi=-Ki*ui;
         
         % save out
         for j=1:length(dofs)
             fem.Boundary.Load.DofId(count)=dofs(j);
             fem.Boundary.Load.Value(count)=Fi(j);
             count=count+1;
         end

      end
              
end


