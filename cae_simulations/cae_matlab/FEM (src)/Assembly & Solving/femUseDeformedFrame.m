% use deformed frame

function fem=femUseDeformedFrame(fem)

% % update nodes
fem.xMesh.Node.Coordinate=fem.Sol.DeformedFrame.Node.Coordinate;
fem.xMesh.Node.Normal=fem.Sol.DeformedFrame.Node.Normal;
fem.xMesh.Node.NormalReset=fem.Sol.DeformedFrame.Node.NormalReset;

% update elements
nele=length(fem.xMesh.Element);

for i=1:nele
    fem.xMesh.Element(i).Tmatrix=fem.Sol.DeformedFrame.Element(i).Tmatrix;
end


% % update fem structure
% 
% % set options
% fem.Options.StiffnessUpdate=true;
% fem.Options.MassUpdate=false;
% fem.Options.ConnectivityUpdate=false; 
% 
% fem=femPreProcessing(fem);