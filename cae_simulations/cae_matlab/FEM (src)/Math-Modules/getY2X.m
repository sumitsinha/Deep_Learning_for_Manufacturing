function Y=getY2X(X)

% use null space
NS=null(X);

Y=NS(:,1); 