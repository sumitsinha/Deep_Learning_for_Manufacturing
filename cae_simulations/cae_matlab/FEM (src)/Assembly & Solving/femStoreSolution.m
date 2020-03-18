% store current data set
function fem=femStoreSolution(fem)

% store:
% 1: solution U
% 2: reaction forces R

% store data set
iddata=fem.Sol.SolId;

fem.Sol.USet.U{iddata}=fem.Sol.U;
fem.Sol.USet.R{iddata}=fem.Sol.R;