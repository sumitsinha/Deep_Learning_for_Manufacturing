function sigmav=getVonMisesField(sigma)

% Sigma=[nnode, 6]; (% [sp_1; sp_2; sp_3])

% Sigmav=[nnode, 1]; 

n=size(sigma,1 );
sigmav=zeros(n,1);

for i=1:n
   si=sigma(i,:);
   
   % save back
   sigmav(i)=sqrt( 1/2*( (si(1)-si(2))^2 + (si(1)-si(3))^2 + (si(2)-si(3))^2 ) );
    
end
