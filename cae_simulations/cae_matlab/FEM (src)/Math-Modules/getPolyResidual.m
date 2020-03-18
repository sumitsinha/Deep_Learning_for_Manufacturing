function polymodel=getPolyResidual(polymodel, Xtest, Ytest)

% Xtest: independent variables [nsample, nvars] 
% Ytest: dependent variables [nsample, 1]

%---
Ym = evalPolyFit(polymodel, Xtest);

% evaluate R^2
SSr = norm(Ytest - Ym)^2;
SSt=norm(Ytest-mean(Ytest))^2;

polymodel.R2 = max(0,1 - SSr/SSt);

% get root mean square error
polymodel.RMSE = sqrt(mean((Ytest - Ym).^2));