% set node loads user

%..
function fem=setLoadUser(fem)

% read number of load conditions
nc=length(fem.Boundary.Load.User);

%-
if nc==0 
    return
end

for i=1:nc
    idof=fem.Boundary.Load.User(i).DofId; % id dofs
    ival=fem.Boundary.Load.User(i).Value; % load value
    
    if size(idof,1)>1
        idof=idof';
    end
    if size(ival,1)>1
        ival=ival';
    end
    % save back
    fem.Boundary.Load.DofId=[fem.Boundary.Load.DofId, idof];
    fem.Boundary.Load.Value=[fem.Boundary.Load.Value, ival];
end 


