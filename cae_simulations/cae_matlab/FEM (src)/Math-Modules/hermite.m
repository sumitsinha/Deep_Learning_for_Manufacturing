% build hermite polynominal (1-d)
function polymodel=hermite(degree)

% INPUT:
% degree: polynomial degree

% OUTPUT
% polymodel
%        .ModelTerms = list of terms in the model
%        .Coefficients = regression coefficients

polymodel.Coefficients = hermiteCoeff(degree)/sqrt(factorial(degree));
polymodel.ModelTerms = getFullPolyModel(degree, 1);

%---------------
function h = hermiteCoeff(degree)
% This is the reccurence construction of a Hermite polynomial, i.e.:
%   H0(x) = 1
%   H1(x) = x
%   H[degree](x) = H[degree-1](x) - (degree-1)*H[degree-2](x)

if 0==degree 
    h = 1;
elseif 1==degree 
    h = [1 0];
else % degree
    
    h1 = zeros(1,degree+1);
    h1(1:degree) = hermiteCoeff(degree-1);
    
    h2 = zeros(1,degree+1);
    h2(3:end) = (degree-1)*hermiteCoeff(degree-2);
    
    h = h1 - h2;
    
end
