function ypred = evalExpFit(expmodel, indepvar)
% Evaluate a exp model as a function of its variables
%
% INPUT
%  indepvar - (n x p) array of independent variables as columns
%        n is the number of data points to evaluate
%        p is the dimension of the independent variable space

%  expmodel - A structure containing a regression model 

% OUTPUT
%  ypred - nx1 vector of predictions through the model.

A=expmodel.Coefficients(1);
B=expmodel.Coefficients(2);

ypred=A.*exp(B.*indepvar);
