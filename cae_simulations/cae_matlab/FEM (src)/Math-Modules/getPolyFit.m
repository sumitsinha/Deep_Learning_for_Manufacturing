% calculate the polynomial fitting
function polymodel = getPolyFit(Xtrain, Ytrain, degree)

% INPUT:
% Xtrain: independent variables [nsample, nvars]
% Ytrain: dependent variables [nsample, 1] 

% degree: polynomial degree

% OUTPUT
% polymodel
%        polymodel.ModelTerms = list of terms in the model
%        polymodel.Coefficients = regression coefficients
%        polymodel.R2 = coefficient of determination (R^2)
%        polymodel.RMSE = Root mean squared error

% Fit the model:
% poly=f(X1, X2,...,Xnvars) + error

% get size of the data set
[nsample, nvars]=size(Xtrain);

% Automatically scale the independent variables to unit variance
stdind = sqrt(diag(cov(Xtrain)));
if any(stdind==0)
  warning('Polynomial fitting: constant variance encountered!')
  stdind(stdind==0) = 1;
end

% scaled variables
Xs = Xtrain*diag(1./stdind);

% build the model
modelterms = getFullPolyModel(degree, nvars);

% build the design matrix
nt = size(modelterms,1);

M = ones(nsample,nt);
scalefact = ones(1,nt);
for i = 1:nt
  for j = 1:nvars
    M(:,i) = M(:,i).*Xs(:,j).^modelterms(i,j);
    scalefact(i) = scalefact(i)/(stdind(j)^modelterms(i,j));
  end
end

% solve the problem
polymodel.ModelTerms = modelterms;
polymodel.Coefficients = (M\Ytrain)'; % least square problem

% recover from scaling
polymodel.Coefficients=polymodel.Coefficients.*scalefact;

% evaluate R^2
Ym=evalPolyFit(polymodel, Xtrain);

SSr = norm(Ytrain - Ym)^2;
SSt=norm(Ytrain)^2; % not centered

R2 = max(0,1 - SSr/SSt);     
RMSE=sqrt(mean((Ytrain - Ym).^2));

%---
polymodel.Degree=degree;
polymodel.R2=R2;
polymodel.RMSE=RMSE;
