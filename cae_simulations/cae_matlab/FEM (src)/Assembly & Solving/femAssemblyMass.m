% assembly mass matrix
function Ma=femAssemblyMass(fem)

% read # of dofs
nTot=fem.Sol.nDoF;

%-----------------------------------
% STEP 1a: assemblying equations
disp('Assemblying sparse pattern:...')

[irow,...
     icol,...
     Xk]=getAssemblySparsityMass(fem); %# OK MEX
 
 % STEP 1b: get assembly matrix
disp('Assemblying sparse mass matrix:...')
Ma=femSparse(irow,...
               icol, ...
               Xk, ...
               nTot, nTot);

