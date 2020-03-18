% Flip Normal of the Selected Component
   
%..
function fem=flipNormalComponent(fem,...
                                 idcomp)


%-
fem.Domain(idcomp).NormalFlip=true;

nele=length(fem.Domain(idcomp).Element);

% flip elements
for i=1:nele
    id=fem.Domain(idcomp).Element(i);
    
    if fem.Options.UseActiveSelection % use selection
          flagactive=fem.Selection.Element.Status(id);
    else
          flagactive=true; % use any element
    end

    if flagactive
    
        tNm=fem.xMesh.Element(id).Tmatrix.Normal;
    
        fem.xMesh.Element(id).Tmatrix.Normal=-tNm;
    end
    
end

%...
nnode=length(fem.Domain(idcomp).Node);

% flip nodes
for i=1:nnode
    
    id=fem.Domain(idcomp).Node(i);
    
    if fem.Options.UseActiveSelection % use selection
          flagactive=fem.Selection.Node.Status(id);
    else
          flagactive=true; % use any element
    end

    if flagactive
        
        tNm=fem.xMesh.Node.Normal(id,:);

        fem.xMesh.Node.Normal(id,:)=-tNm;
    
    end
    
end


