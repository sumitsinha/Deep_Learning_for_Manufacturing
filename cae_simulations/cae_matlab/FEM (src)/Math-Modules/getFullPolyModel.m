function modelterms = getFullPolyModel(degree, p)
% 
% arguments: (input)
%  degree - scalar integer, defines the total (maximum) degree 
%
%  p     - scalar integer - defines the dimension of the
%          independent variable space
%
% arguments: (output)
%  modelterms - exponent array for the model
 
% build the exponent array recursively
if p == 0
  % terminal case
  modelterms = [];
elseif (degree == 0)
  % terminal case
  modelterms = zeros(1,p);
elseif (p==1)
  % terminal case
  modelterms = (degree:-1:0)';
else
  % general recursive case
  modelterms = zeros(0,p);
  for k = degree:-1:0
    t = getFullPolyModel(degree-k,p-1);
    nt = size(t,1);
    modelterms = [modelterms;[repmat(k,nt,1),t]];
  end
end
