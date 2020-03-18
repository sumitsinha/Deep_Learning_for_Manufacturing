function A=ivech(h, n)
% A = ivech(h, n)
% h is the column vector of elements on or below the main diagonal of A (lower triangular part of A).
% A will be square and symmetric (nxn)

count=1;
value=1;
A=zeros(n,n);
for j=1:n
    for i=1:n
        if i==j || i>j
            A(value)=h(count);
            count=count+1;
        end
        
        A(j,i)=A(i,j);
        
        value=value+1;
    end
end
