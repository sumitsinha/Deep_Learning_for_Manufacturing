function v = ns(A)
% NS Solve the null-space problem A*v=0 
%
% NS return the 1-d null-space of A 

[~, D, V] = svd(A);
D = diag(D);

% check condition number
c = D(1)/D(end-1);
if c > 200
  warning('Null Space: Condition number is %0.f',c)
end

% Notice that the last column of V corresponds to the solution vector in the least square sense
v=V(:,end);


