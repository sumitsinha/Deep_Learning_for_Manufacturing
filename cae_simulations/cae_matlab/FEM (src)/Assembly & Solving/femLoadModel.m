function fem=femLoadModel(fem)

% load:
% 1: boundary conditions
% 2: domain conditions

iddata=fem.Sol.SolId;

% store boundary
fem.Boundary=fem.Sol.ModelSet.Boundary{iddata};

% store domain
fem.Domain=fem.Sol.ModelSet.Domain{iddata};