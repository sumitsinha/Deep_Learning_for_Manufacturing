% reset FEM
function fem=femReset(fem)

% reset boundary
fem=femResetBoundary(fem);

% reset solution
fem=femResetSolution(fem);
    