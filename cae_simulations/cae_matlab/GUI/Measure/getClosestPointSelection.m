function [mdist,idSelected]=getClosestPointSelection(ax, vertex)

Pc=get(ax,'CameraPosition'); %- camera position
Pt=get(ax,'CameraTarget');  %- camera target
Vup=get(ax,'CameraUpVector'); %- camera up-vector

Pp=get(ax,'CurrentPoint'); %- picked point
Pp=Pp(1,:);

%- built frame 
Zc=(Pc-Pt)/norm(Pc-Pt);
Vup=Vup/norm(Vup);

Xc=cross(Vup,Zc);
Xc=Xc/norm(Xc);
Yc=cross(Zc,Xc);
Yc=Yc/norm(Yc);

R = [Xc' Yc' Zc']; 

% get closted point
vertexr=R'*vertex';
Pp=R'*Pp';

%- take just (x-y) components
vertexr= vertexr(1:2,:);
Pp=Pp(1:2);

%- calculate distances
diff=[vertexr(1,:)-Pp(1); vertexr(2,:)-Pp(2)];
dist=sqrt(sum(diff.^2,1));

%- finally, find-out the index related to the minimum distance
[mdist,idSelected]=min(dist);
