% get no. of terms of the expansion
function N=polChaosNoTerms(degree, d)

% INPUT:
% degree: polynomial degree
% d: no. of independent variables

% OUTPUT:
% N: minimum no. of terms of chaos expansion

N=factorial(degree+d)/(factorial(degree) * factorial(d));
