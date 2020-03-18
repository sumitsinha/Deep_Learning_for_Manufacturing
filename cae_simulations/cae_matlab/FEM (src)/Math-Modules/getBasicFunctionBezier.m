function b=getBasicFunctionBezier(t,n)

    % t: curve parameter
    % n: curve degree
    
    % b: [1, n+1] basic functions
    
b=zeros(1,n+1);

% loop over n
for i=0:n
    bfact=factorial(n)/(factorial(i) * factorial(n-i) );
    b(i+1)=bfact*t^i*(1-t)^(n-i);
end
    
    
    
