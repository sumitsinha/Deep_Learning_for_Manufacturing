% set node loads user (It is used by the FAO module for modelling "release")

%..
function fem=setLoadUser(fem)

% read number of load conditions
nc=length(fem.Boundary.Load.User);

%-
if nc==0 
    return
end

for i=1:nc
    idom=fem.Boundary.Load.User(i).Domain; % id dofs
    if ~isempty(idom)
        ival=fem.Boundary.Load.User(i).Value; % load value

        nd=length(idom);
        idof=[];
        for j=1:nd
            inodej=fem.Domain(idom(j)).Node;
            idofj=fem.xMesh.Node.NodeIndex(inodej,:);
            idof=[idof, idofj(:)']; %#ok<AGROW>
        end
        idof=sort(idof);

        % save back
        fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId, idof];
        fem.Boundary.Load.Value=[fem.Boundary.Load.Value, ival];
    end
end 


