% assembly stiffness matrix
function Ka=femAssemblyStiffness(fem)

% read # of dofs
nTot=fem.Sol.nDoF;

%-----------------------------------
% STEP 1a: assemblying equations
disp('Assemblying sparse pattern:...')

[irow,...
     icol,...
     Xk,...
     ~]=getAssemblySparsityStiffness(fem); %# OK MEX
 
 % STEP 1b: get assembly matrix
disp('Assemblying sparse stiffness matrix:...')

Ka=femSparse(irow,...
               icol, ...
               Xk, ...
               nTot, nTot);

