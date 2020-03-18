% build hermite polynominal in n-d space
function y=hermiten(degree, d, iHe, X)

% INPUT:
% degree: polynomial degree
% d: no. of independent variables
% iHe: polynomial ID
% X: independent vector (m, d) - normal distributed (mean=0; std=1)

% OUTPUT
% y: evaluation of polynomial at X (m, 1)
%-----

% STEP 1: get terms of the polynomial
modelterms = getFullPolyModel(degree, d);

%--
% swap first and last terms
t=modelterms(end,:);
modelterms(end,:)=modelterms(1,:);
modelterms(1,:)=t;
%--

% STEP 2: evaluation polynomial

% no. of points
m=size(X,1);

y=ones(m,1);
for j=1:d
    t=hermite(modelterms(iHe, j));
    y=y.*evalPolyFit(t, X(:,j));
end