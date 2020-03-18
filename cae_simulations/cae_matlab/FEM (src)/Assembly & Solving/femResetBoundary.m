% reset FEM boundary
function fem=femResetBoundary(fem)

% SPC conditions (only LINEAR constraints)
fem.Boundary.Constraint.SPC.Id=[]; % id dofs constrained
fem.Boundary.Constraint.SPC.Value=[];

% loads
fem.Boundary.Load.DofId=[];
fem.Boundary.Load.Value=[];

% equations
fem.Boundary.Constraint.MPC=[];
fem.Boundary.Constraint.ContactMPC=[];
fem.Boundary.Constraint.ContactWset=[];



