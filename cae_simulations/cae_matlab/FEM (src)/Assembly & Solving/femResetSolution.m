% reset FEM solution
function fem=femResetSolution(fem)

fem.Sol.Kast.n=fem.Sol.Kast.ndofs;

fem.Sol.nLSPC=0; % # of Lagrange multiplier for SPC (bilateral)
fem.Sol.nLMPC=0; % # of Lagrange multiplier for MPC (bilateral)
fem.Sol.nLCt=0; % # of Lagrange multiplier for contact (unilateral)

fem.Sol.U=[]; % all set
fem.Sol.res=[]; % numerical residuals
fem.Sol.R=[]; % reaction forces (all set)

fem.Sol.Gap=[];
fem.Sol.Lamda=[];

