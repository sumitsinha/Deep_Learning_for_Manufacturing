function fem=femStoreModel(fem)

% store:
% 1: boundary conditions
% 2: domain conditions

iddata=fem.Sol.SolId;

% store boundary
fem.Sol.ModelSet.Boundary{iddata}=fem.Boundary;

% store domain
fem.Sol.ModelSet.Domain{iddata}=fem.Domain;