% plot load conditions
function loadPlot(fem)

L=fem.Post.Options.LengthAxis;

% read number of loads
nc=length(fem.Boundary.Load.Node);

Pm=[];
Nm=[];
for i=1:nc
     
     etype=fem.Boundary.Load.Node(i).Type;
     
     id=fem.Boundary.Load.Node(i).Node;
     dofs=fem.Boundary.Load.Node(i).DoF;
     value=fem.Boundary.Load.Node(i).Value;

         for j=1:length(id)
             
             if ~strcmp(etype{j},'not-assigned')

                 for z=1:length(dofs)
                    P0=fem.xMesh.Node.Coordinate(id(j),:);

                    if dofs(z)==1 || dofs(z)==4
                        N0=[1 0 0]*sign(value(z));
                    elseif dofs(z)==2 || dofs(z)==5
                        N0=[0 1 0]*sign(value(z));
                    elseif dofs(z)==3 || dofs(z)==6
                        N0=[0 0 1]*sign(value(z));
                    end

                    Pm=[Pm;P0];
                    Nm=[Nm;N0];

                 end
             end
         end
end

% read number of loads (element-based)
nc=length(fem.Boundary.Load.Element);

for i=1:nc
    
    etype=fem.Boundary.Load.Element(i).Type;
     
    if ~strcmp(etype,'not-assigned')
         
         ref=fem.Boundary.Load.Element(i).Reference; % reference

         value=fem.Boundary.Load.Element(i).Value;

        if strcmp(ref,'cartesian')

            dofs=fem.Boundary.Load.Element(i).DoF;
            
            for z=1:length(dofs)
                P0=fem.Boundary.Load.Element(i).Pm;

                if dofs(z)==1 || dofs(z)==4
                    N0=[1 0 0]*sign(value(z));
                elseif dofs(z)==2 || dofs(z)==5
                    N0=[0 1 0]*sign(value(z));
                elseif dofs(z)==3 || dofs(z)==6
                    N0=[0 0 1]*sign(value(z));
                end

                Pm=[Pm;P0];
                Nm=[Nm;N0];
            end



        elseif strcmp(ref,'vectorTra') || strcmp(ref,'vectorRot')

            P0=fem.Boundary.Load.Element(i).Pm;
            N0=fem.Boundary.Load.Element(i).Nm;

            Pm=[Pm;P0];
            Nm=[Nm;N0];

        end
    
    end

end

% plot arrow
if ~isempty(Pm)
    quiver3(Pm(:,1),Pm(:,2),Pm(:,3),Nm(:,1),Nm(:,2),Nm(:,3),L,...
            'color','b',...
            'linewidth',1,...
            'parent',fem.Post.Options.ParentAxes);
end   
   
if fem.Post.Options.ShowAxes
    set(fem.Post.Options.ParentAxes,'visible','on')
else
    set(fem.Post.Options.ParentAxes,'visible','off')
end

