% load data set
function fem=femLoadSolution(fem)

% load:
% 1: solution U
% 2: reaction forces R

% load data set
iddata=fem.Sol.SolId;

fem.Sol.U=fem.Sol.USet.U{iddata};
fem.Sol.R=fem.Sol.USet.R{iddata};

% Calculating deformed frame...
fem=getDeformedFrame(fem);

% Calculating gap variables
fem=getGapsVariable(fem);

