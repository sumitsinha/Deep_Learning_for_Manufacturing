function ypred = evalPolyFit(polymodel, indepvar)
% Evaluate a polynomial model as a function of its variables
%
% INPUT
%  indepvar - (n x p) array of independent variables as columns
%        n is the number of data points to evaluate
%        p is the dimension of the independent variable space

%  polymodel - A structure containing a regression model from polyfitn

% OUTPUT
%  ypred - nx1 vector of predictions through the model.

% get size of the data set
[n,p]=size(indepvar);

% evaluate the model
nt = size(polymodel.ModelTerms,1);
ypred = zeros(n,1);
for i = 1:nt
  t = ones(n,1);
  for j = 1:p
    t = t.*indepvar(:,j).^polymodel.ModelTerms(i,j);
  end
  ypred = ypred + t*polymodel.Coefficients(i);
end


