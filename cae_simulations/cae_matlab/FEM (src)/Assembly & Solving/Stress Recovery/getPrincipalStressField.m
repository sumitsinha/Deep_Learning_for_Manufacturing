% get principal stress fields
function sigmap=getPrincipalStressField(sigma)

% Sigma=[nnode, 6]; (% [s_x; s_y; s_z; s_xy; s_xz; s_yz])

% Sigmap=[nnode, 3]; (% [sp_1; sp_2; sp_3])

n=size(sigma,1 );
sigmap=zeros(n,3);

for i=1:n
   si=sigma(i,:);
   
   % build tensor matrix
   stensor=[si(1) si(4) si(5)
            si(4) si(2) si(6)
            si(5) si(6) si(3)];
        
   % get eigenvalues (principal values)
   sp=eig(stensor);
   
   % save back
   sigmap(i,:)=sp;
   
end
