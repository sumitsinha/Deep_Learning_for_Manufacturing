% calculate natural coordinate from real coordinate
function [csi,eta]=mapxy2csietaQuad4(x,y,P)

% Partially inspired on:
%...................................................................................
% http://www.colorado.edu/engineering/cas/courses.d/IFEM.d/IFEM.Ch23.d/IFEM.Ch23.pdf
%...................................................................................


% get coordinates
P=P([3 4 1 2],:); % index have been swaped to match with the numbering of shape functions
xp=P(:,1); x1=xp(1); x2=xp(2); x3=xp(3); x4=xp(4);
yp=P(:,2); y1=yp(1); y2=yp(2); y3=yp(3); y4=yp(4);


xccsi=x1+x2-x3-x4;
xceta=x1-x2-x3+x4;
xb=x1-x2+x3-x4;

yccsi=y1+y2-y3-y4;
yceta=y1-y2-y3+y4;
yb=y1-y2+y3-y4;

x0=(x1+x2+x3+x4)/4;

y0=(y1+y2+y3+y4)/4;

J0=(x3-x1)*(y4-y2)-(x4-x2)*(y3-y1);
A=J0/2;

J1=(x3-x4)*(y1-y2)-(x1-x2)*(y3-y4);

J2=(x2-x3)*(y1-y4)-(x1-x4)*(y2-y3);

dx=x-x0;
dy=y-y0;

bcsi=A-dx*yb+dy*xb;
beta=-A-dx*yb+dy*xb;

ccsi=dx*yccsi-dy*xccsi;
ceta=dx*yceta-dy*xceta;

dcsi=-sqrt(bcsi^2-2*J1*ccsi)-bcsi;
deta=sqrt(beta^2+2*J2*ceta)-beta;

csi=2*ccsi/dcsi;

eta=2*ceta/deta;







                        % % get constants
                        % x01=x(1)+x(2)+x(3)+x(4);
                        % x02=x(1)+x(2)-x(3)-x(4);
                        % x03=x(1)-x(2)+x(3)-x(4);
                        % x04=x(1)-x(2)-x(3)+x(4);
                        % 
                        % x05=4*xp-x01;
                        % 
                        % y01=y(1)+y(2)+y(3)+y(4);
                        % y02=y(1)+y(2)-y(3)-y(4);
                        % y03=y(1)-y(2)+y(3)-y(4);
                        % y04=y(1)-y(2)-y(3)+y(4);
                        % 
                        % y05=4*yp-y01;
                        % 
                        % Acsi=y04*x03-x04*y03;
                        % Bcsi=x05*y03-x04*y02-y05*x03+y04*x02;
                        % Ccsi=x05*y02-y05*x02;
                        % 
                        % Aeta=y02*x03-x02*y03;
                        % Beta=x05*y03-x02*y04-y05*x03+y02*x04;
                        % Ceta=x05*y04-y05*x04;
                        % 
                        % % solve for csi
                        % if Acsi<=eps
                        %     csi=-Ccsi/Bcsi;
                        % else
                        %     tmp(1)=(-Bcsi+sqrt(Bcsi^2-4*Acsi*Ccsi))/(2*Acsi);
                        %     tmp(2)=(-Bcsi-sqrt(Bcsi^2-4*Acsi*Ccsi))/(2*Acsi);
                        %     
                        %     if abs(tmp(1))>abs(tmp(2)) % take solution nearest to zero
                        %         csi=tmp(2);
                        %     else
                        %         csi=tmp(1);
                        %     end 
                        % end
                        % 
                        % % solve for eta
                        % if Aeta<=eps
                        %     eta=-Ceta/Beta;
                        % else
                        %     tmp(1)=(-Beta+sqrt(Beta^2-4*Aeta*Ceta))/(2*Aeta);
                        %     tmp(2)=(-Beta-sqrt(Beta^2-4*Aeta*Ceta))/(2*Aeta);
                        %     
                        %     if abs(tmp(1))>abs(tmp(2)) % take solution nearest to zero
                        %         eta=tmp(2);
                        %     else
                        %         eta=tmp(1);
                        %     end 
                        % end

