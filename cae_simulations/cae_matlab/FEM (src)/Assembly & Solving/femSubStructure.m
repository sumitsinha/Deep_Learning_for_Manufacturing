% this function estract a sub structure from the source model "fems" and pass to the destination "femd"
function femd=femSubStructure(fems, idele)

% fems: source model
% idele: list of element to extract

% femd: destination model

fprintf('Model extraction:\n');

% remove duplicates
fprintf('    Remove duplicates elements...\n');
idele=unique(idele);

% extract elements
fprintf('    Extract elements...\n');
nele=length(idele);

% get counters
cq=0;
ct=0;
for i=1:nele
    te=fems.xMesh.Element(idele(i)).Type;
    
    if strcmp(te,'quad')
        cq=cq+1;
    elseif strcmp(te,'tria')
        ct=ct+1;
    end
end

% update elements
quads=zeros(cq,4);
quads(1)=-1;
trias=zeros(ct,3);
trias(1)=-1;

cq=1;
ct=1;
for i=1:nele
    e=fems.xMesh.Element(idele(i)).Element;
    te=fems.xMesh.Element(idele(i)).Type;
    
    if strcmp(te,'quad')
        quads(cq,:)=e;
        cq=cq+1;
    elseif strcmp(te,'tria')
        trias(ct,:)=e;
        ct=ct+1;
    end
end

% extract node
fprintf('    Extract nodes...\n');
idnode=[];
if quads(1)~=-1
    idnode=[idnode
            quads(:)];
end

if trias(1)~=-1
    idnode=[idnode
            trias(:)];
end

idnode=unique(idnode);
          
node=fems.xMesh.Node.Coordinate(idnode,:);

% renumber elements ids
fprintf('    Renumber elements...\n');
if quads(1)~=-1
    quads=renumberElements(quads, idnode);
end

if trias(1)~=-1
    trias=renumberElements(trias, idnode);
end

% init new fem strucure
fprintf('    Initialise destination structure...\n');
femd=femInit();

% save all
femd=femSaveMesh(femd, quads, trias, node);

%---
fprintf('Model extraction completed!\n');
%---------------------------

