function R=vector2Rotation(N)

% build ref. frame
NS=null(N);
x=NS(:,1);
y=NS(:,2);

R=[x, y, N'];

if det(R)<0
    R(:,2)=-R(:,2);
end