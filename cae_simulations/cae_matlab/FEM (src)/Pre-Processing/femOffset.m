% create solid mesh
function [Vertex, Faces]=femOffset(fem, idpart, offValue)

id=fem.Domain(idpart).Node;

node=fem.xMesh.Node.Coordinate(id,:);
normal=fem.xMesh.Node.Normal(id,:);

% apply offset
node=node+offValue*normal;

% close along boundaries
closefaces=[];
for I=1:length(edgeLoop)
    closefaces=[closefaces;calculateClosedFace(edgeLoop{I},nVert)];
end

%%% aggiorna i vertici...
Vertex=[Vertex;VertexOff];

%%% aggiorna le facce...
Faces=[Faces;
       Faces+nVert]; %% aggiorna il contatore...

%%% aggiorna con la chiusura...
Faces=[Faces;
       closefaces];
      

function closefaces=calculateClosedFace(edgeLoop,Nvert)
%%% calcola le facce per la chiusra della mesh di offset...

%%% inizializza...
closefaces=[];
for I=1:length(edgeLoop)-1 %%% numero di vertici sul bordo...
    closefaces=[closefaces;
                edgeLoop(I) edgeLoop(I+1) edgeLoop(I)+Nvert;
                edgeLoop(I+1) edgeLoop(I+1)+Nvert edgeLoop(I)+Nvert];
end

