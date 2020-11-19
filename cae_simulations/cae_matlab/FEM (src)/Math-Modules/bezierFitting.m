function Pc=bezierFitting(Q, n)
    % Q: key points
    % n: curve degree
    
    % Pc: control points

% no. of key points
nq=size(Q,1)-1;

% get point parameters ("chord length alghoritm")
dt=0;
d=zeros(1,n);
for i=1:nq
    d(i)=norm(Q(i+1,:)-Q(i,:));
    dt=d(i)+dt;
end

% normalize into [0, 1]
t=zeros(1,n+1);
t(1)=0;
for i=2:nq
    t(i)=t(i-1)+d(i-1)/dt;
end
t(nq+1)=1;

% inizializza la matrice dei coefficienti
A=zeros(nq+1,n+1);

% fill A rows
for i=1:nq+1
    b=getBasicFunctionBezier(t(i),n);   
    A(i,:)=b;
end

% calcola i punti di controllo incogniti
Pc=A\Q; % LU decomposition


