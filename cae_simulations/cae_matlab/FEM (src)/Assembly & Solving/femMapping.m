function femDst=femMapping(femSrc, femDst)

% INPUT:
    % femSrc: source model
    % femDst: destination model
    
% OUTPUT:
    % femDst: destination model updated
    
% N.B.: gap interpolation is not supported so far.

fprintf('-->>\n');
fprintf('Computing mapping...\n');

% get inputs
domainsrc=femDst.Mapping.Source;
domaindst=femDst.Mapping.Destination;
varMp=femDst.Mapping.MapVariable;
sdistance=femDst.Mapping.SearchDist;

fprintf('    destination: %g\n',domaindst);
fprintf('    source: %g\n',domainsrc);
fprintf('    mapping field: %s\n',varMp);

%------------------------------------------------------------------------  
% STEP 1

% set parameters for interpolation
femSrc.Post.Interp.InterpVariable=varMp;
femSrc.Post.Interp.Domain=domainsrc;
femSrc.Post.Interp.SearchDist=sdistance;

% STEP 2

% get destination data
% get node ids
idnodedst=femDst.Domain(domaindst).Node;

% get nodes
nodedst=femDst.xMesh.Node.Coordinate(idnodedst,:);
femSrc.Post.Interp.Pm=nodedst;

% STEP 3

% run interpolation
fprintf('    dest-->>source interpolation:\n');
femSrc=getInterpolationData(femSrc);

% STEP 4

% get interpolation data
dataint=femSrc.Post.Interp.Data;

% update destination
fprintf('    update destination domain:\n');
if strcmp('u',varMp) % U field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,1))=dataint;
    
elseif strcmp('v',varMp) % V field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,2))=dataint;
     
elseif strcmp('w',varMp) % W field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,3))=dataint;
    
elseif strcmp('alfa',varMp) % alfa field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,4))=dataint;

elseif strcmp('beta',varMp) % beta field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,5))=dataint;

elseif strcmp('gamma',varMp) % gamma field
    
    index=getIndexNode(idnodedst,6);
    femDst.Sol.U(index(:,6))=dataint;
    
elseif strcmp('user',varMp) % user field
    
     % update user expression
     nt=size(femDst.xMesh.Node.Coordinate,1);
     femDst.Sol.UserExp=zeros(nt,1);
     
     femDst.Sol.UserExp(idnodedst)=dataint;
        
end

fprintf('-->>\n');

