function initGCS(data)

lsymbol=data.Axes3D.Options.LengthAxis;
plotFrame(eye(3,3), [0 0 0], data.Axes3D.AxesGCS, lsymbol,'tempGCS')

set(data.Axes3D.AxesGCS,'xlim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);
set(data.Axes3D.AxesGCS,'ylim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);
set(data.Axes3D.AxesGCS,'zlim',[-lsymbol/2-0.1*lsymbol lsymbol+0.1*lsymbol]);

% set isoview
view(data.Axes3D.AxesGCS, 3)
