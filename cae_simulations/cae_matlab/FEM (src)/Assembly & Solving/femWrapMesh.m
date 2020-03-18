% wrap mesh model based on planarity criteria
function idelenew=femWrapMesh(fem, idele, sharpangle) 

% fem: fem structure
% idele: list of elements
% sharpangle: sharpe angle threshhold to check the planarity 

% idelenew: updated list of elements

% remove duplicates
idele=unique(idele);

% loop over elements
nele=length(idele);
idelenew=[];
for k=1:nele
     idk=idele(k);
     
     idchildk=fem.Sol.Element2Element{idele(k)};   
     
     idchildk=checkPattern(fem, idk, idchildk, sharpangle);
     
     idelenew=[idelenew, idchildk];
end

% remove duplicates
idelenew=unique(idelenew);


% check connectivity 
function idchild=checkPattern(fem, id, idchild, sharpangle)

n=length(idchild);

Nid=fem.xMesh.Element(id).Tmatrix.Normal;

for i=1:n
    Ni=fem.xMesh.Element(idchild(i)).Tmatrix.Normal;
    
    % get angle
    angle=acos(dot(Ni, Nid))*180/pi;
    
    % check the planarity
    if angle>sharpangle
        idchild(i)=0;
    end  
end

idchild(idchild==0)=[];



