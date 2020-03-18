function dV=getBoundingVolume(nodes)

eps=1e-1;

% nodes: xyz coordinates

% set intial fields
dV=initInputDatabase('Selection');

dV.Pm=mean(nodes);
dV.Nm1=[1 0 0];
dV.Nm2=[0 1 0];

for i=1:3
    rmi=max(nodes(:,i)) - min(nodes(:,i));
    dV.Rm(i)=rmi;
end

% save building
rm=mean(dV.Rm);
for i=1:3
    if dV.Rm(i)<=eps
        dV.Rm(i)=rm;
    end
end

