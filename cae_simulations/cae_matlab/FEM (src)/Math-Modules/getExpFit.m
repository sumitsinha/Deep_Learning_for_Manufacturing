% calculate the exp fitting
function expfit = getExpFit(Xtrain, Ytrain)

% INPUT:
% Xtrain: independent variables [nsample, nvars]
% Ytrain: dependent variables [nsample, 1] 

% OUTPUT
% expmodel
%        expmodel.Coefficients = regression coefficients
%        expmodel.R2 = coefficient of determination (R^2)
%        expmodel.RMSE = Root mean squared error

% Fit the model:
% expfit=A*exp(B*x)

% solve the problem: ln(y)=ln(A)+B*x;
Ytrain_log=log(Ytrain);
polymodel = getPolyFit(Xtrain, Ytrain_log, 1);

B=polymodel.Coefficients(1);
A=exp(polymodel.Coefficients(2));

expfit.Coefficients = [A, B];

% evaluate R^2
Ym=evalExpFit(expfit, Xtrain);

SSr = norm(Ytrain - Ym)^2;
SSt=norm(Ytrain)^2; % not centered

R2 = max(0,1 - SSr/SSt);     
RMSE=sqrt(mean((Ytrain - Ym).^2));

%---
expfit.R2=R2;
expfit.RMSE=RMSE;
