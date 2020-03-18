function Area=getAreaTria(P)

xv=P(:,1);
yv=P(:,2);

Area=abs((xv(3)-xv(1))*(yv(2)-yv(1))-(yv(3)-yv(1))*(xv(2)-xv(1)))/2;