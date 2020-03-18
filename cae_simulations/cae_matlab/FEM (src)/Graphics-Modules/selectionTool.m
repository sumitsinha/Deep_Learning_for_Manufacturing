% selection tool (set ACTIVE node/elements)
function selectionTool(fem, idpart)

%--
% pre-requirement: plot has to be active ("figure" and "axes" have been already loaded)  
%--

% create new menu for selection
handle.fem=fem;
handle.idpart=idpart;
handle.Selection=[];

guidata(gcf,handle)

%--
f=uimenu('Label','Selection TOOL');

uimenu(f,'Label','Select (Draw a rectangle)','Callback',{@ROIRectangle, gcf, gca});


% enable selection on rectangle selection
function ROIRectangle(src, event, fig, ax)

warning off

% inizialize selection phase
set(fig,'WindowButtondownFcn',{@Sclick, fig, ax});


function Sclick(src,event,fig,ax)

Pc=get(ax,'CameraPosition'); %- camera position
Pt=get(ax,'CameraTarget');  %- camera target
Vup=get(ax,'CameraUpVector'); %- camera up-vector

P1=get(ax,'CurrentPoint'); %- picked point
P1=P1(1,:);

%- built frame 
Zc=(Pc-Pt)/norm(Pc-Pt);
Vup=Vup/norm(Vup);

Xc=cross(Vup,Zc);
Xc=Xc/norm(Xc);
Yc=cross(Zc,Xc);
Yc=Yc/norm(Yc);

R = [Xc;Yc;Zc];  

XL=get(ax,'XLim');
YL=get(ax,'YLim');
ZL=get(ax,'ZLim');

%- start mouse motion
set(fig,'WindowButtonMotionFcn',{@moveMouse,P1,fig,ax,R,XL,YL,ZL})

function moveMouse(src,event,P1,fig,ax,R,XL,YL,ZL)

set(ax,'XLim',XL,'YLim',YL,'ZLim',ZL);

%- actual mouse position
P4 = get(ax,'CurrentPoint');
P4=P4(1,:);

%- tranform into local frame
P1=R*P1';
P1(3)=0;

P4=R*P4';
P4(3)=0;

%- built rectangle selection
P2=[P1(1) P4(2) 0];
P3=[P4(1) P1(2)  0];

%- ...
Vertex=[P1';P2;P4';P3];

%- go-back to global frame
Vertext=R'*Vertex';

% Plotta il rettangolo:
PreRect=findobj(fig,'tag','slt');
delete(PreRect);

%- plot rectangle
patch('Faces',[1 2 3 4],'Vertices',Vertext','LineStyle',':','facecolor','g','facealpha',0.5,...
      'parent',ax,...
      'linewidth',2,'tag','slt');

%-call mouse buttonUp
set(fig,'WindowButtonUpFcn',{@endClick, fig, R, Vertex});


function endClick(src,event, fig, R, Vertex)

%--
handle=guidata(fig);

fem=handle.fem;
idpart=handle.idpart;
%--

% set initial selection condition
nnode=size(fem.xMesh.Node.Coordinate,1);
fem.Selection.Node.Status=false(1,nnode);

%--
nele=length(fem.xMesh.Element);
fem.Selection.Element.Status=false(1,nele);

%- transforms points
idnode=[fem.Domain(idpart).Node];
nodes=fem.xMesh.Node.Coordinate(idnode,:);

Ncoordt=R*nodes';

Vertex=Vertex(:,1:2); %- only x-y

% check points inside the selection rectangle
inPol=inpolygon(Ncoordt(1,:),Ncoordt(2,:),Vertex(:,1),Vertex(:,2));

% update structure
idnode=idnode(inPol);

%--
%--------------------------------------------------------------------------

% disable all controls
set(fig,'WindowButtondownFcn','');
set(fig,'WindowButtonMotionFcn','');
set(fig,'WindowButtonUpFcn','');

% save structure
handle.fem=fem;
handle.Selection=idnode;
guidata(fig,handle);

