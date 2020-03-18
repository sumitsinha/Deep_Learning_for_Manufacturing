function femDst=femSubModel(femSrc, femDst)

% INPUT:
    % femSrc: source model
    % femDst: destination model
    
% OUTPUT:
    % femDst: destination model updated

%-------------
% N.B.: it is assumed that the domains on the source and destination structure are perfectly matching
%-------------

fprintf('-->>\n');
fprintf('Running sub-model...\n');

% re-set node constraints
femDst.Boundary.Constraint.Bilateral.Node=[];

% no. of domains
nd=femSrc.Sol.nDom;

% get elements and nodes for destination domain
nnode=size(femDst.xMesh.Node.Coordinate,1);

% run over all domains
for i=1:nd
   
    sdistance=femDst.Domain(i).SubModel.SearchDist;
    
    % STEP 1: get cutting edges
    fprintf('    working on domain: %g\n',i);
    
    fprintf('    getting cut edges:\n');

    % get boundary points:
    pbi=femDst.Domain(i).SubModel.CuttingSt;
    
    cutids=boundary3Nodes(pbi, femDst.Denoise.Tria, nnode);
        
    % STEP 2: run interpolation
    fprintf('    interpolating cut edges:\n');
    
    % set parameters for interpolation
    
    ncut=length(cutids);
    cutval=zeros(ncut,6); % u, v, w, alfa, beta, gamma.
    
    femSrc.Post.Interp.Domain=i;
    femSrc.Post.Interp.SearchDist=sdistance;
    femSrc.Post.Interp.Pm=femDst.xMesh.Node.Coordinate(cutids,:);
    
    % U field
    femSrc.Post.Interp.InterpVariable='u';
    femSrc=getInterpolationData(femSrc);

    cutval(:,1)=femSrc.Post.Interp.Data;

    % V field
    femSrc.Post.Interp.InterpVariable='v';
    femSrc=getInterpolationData(femSrc);

    cutval(:,2)=femSrc.Post.Interp.Data;
    
    % W field
    femSrc.Post.Interp.InterpVariable='w';
    femSrc=getInterpolationData(femSrc);

    cutval(:,3)=femSrc.Post.Interp.Data;
    
    % alfa field
    femSrc.Post.Interp.InterpVariable='alfa';
    femSrc=getInterpolationData(femSrc);

    cutval(:,4)=femSrc.Post.Interp.Data;
    
    % beta field
    femSrc.Post.Interp.InterpVariable='beta';
    femSrc=getInterpolationData(femSrc);

    cutval(:,5)=femSrc.Post.Interp.Data;
    
    % gamma field
    femSrc.Post.Interp.InterpVariable='gamma';
    femSrc=getInterpolationData(femSrc);

    cutval(:,6)=femSrc.Post.Interp.Data;
    
    % STEP 3: apply boundary constraints
    fprintf('    applying boundary constraints at cut edges:\n');
    
    count=1;
    for k=1:ncut
        
        for j=1:6
            femDst.Boundary.Constraint.Bilateral.Node(count).Node=cutids(k);
            femDst.Boundary.Constraint.Bilateral.Node(count).Reference='cartesian';
            femDst.Boundary.Constraint.Bilateral.Node(count).DoF=j;
            femDst.Boundary.Constraint.Bilateral.Node(count).Value=cutval(k,j);
            femDst.Boundary.Constraint.Bilateral.Node(count).Physic='shell';
            count=count+1;
        end
        
    end
    
    % save structure
    femDst.Domain(i).SubModel.CuttingId=cutids;
    
end

fprintf('-->>\n');



