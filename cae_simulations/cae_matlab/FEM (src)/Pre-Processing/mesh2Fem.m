function varargout=mesh2Fem(fem, datainput, opt)

% datainput: {filename} => opt='file'
% datainput: .Node; .Quad; .Tria => opt='geom'

if nargin==2
    opt='file';
end

%------------------------
% no. of mesh data
ndata=length(datainput);

% set intial couters
nDom=1;
nnodes=0;
nquads=0;
ntrias=0;

% set initial fields

% quads
fem.Sol.Quad.Count=0;
fem.Sol.Quad.Id=[];
fem.Sol.Quad.Element=[];

% trias
fem.Sol.Tria.Count=0;
fem.Sol.Tria.Id=[];
fem.Sol.Tria.Element=[];

% nodes
fem.xMesh.Node.Coordinate=[];
fem.xMesh.Node.CoordinateReset=[];
fem.xMesh.Node.Component=[];
fem.xMesh.Node.Normal=[];
fem.xMesh.Node.NormalReset=[];
fem.xMesh.Node.NodeIndex=[];
fem.xMesh.Node.Tnode=[];

fem.Sol.DeformedFrame.Node.Coordinate=[];
fem.Sol.DeformedFrame.Node.Normal=[];
fem.Sol.DeformedFrame.Node.NormalReset=[];

% loop over all files
for kf=1:ndata

    if strcmp(opt,'file')
        
        filename=datainput{kf};

        if ~isempty(filename)

            disp('>>---')
            fprintf('Reading mesh: %s\n ', filename)
            disp('>>---')

            % get file extension
            [~,~,ext]=fileparts(filename);

            % abaqus
            if strcmp(ext,'.inp')
                disp('Reading Abaqus Mesh File...')
                [quad,tria,node]=readMeshAbaqus(filename); % # MEX function

            % nastran
            elseif strcmp(ext,'.bdf') || strcmp(ext,'.dat')
                disp('Reading Nastran Mesh File...')
                [quad,tria,node]=readMeshNas(filename); % # MEX function

            % stl format
            elseif strcmp(ext,'.stl') || strcmp(ext,'.STL') 
                disp('Reading STL Mesh File...')

                stltype = getStlFormat(filename);

                if strcmp(stltype,'ascii')

                    quad=-1; % only trias elements are supported
                    [tria, node]=readMeshStlAscii(filename); % matlab function

                elseif strcmp(stltype,'binary')

                    quad=-1; % only trias elements are supported
                    [tria, node]=readMeshStlBin(filename); % matlab function

                end

            else
                error('Mesh file not reconognised @ "%s"', filename);
            end 

            % store data
            [fem, nDom, nnodes, nquads, ntrias]=localStoreData(fem, nDom, node, quad, tria, nnodes, nquads, ntrias);

        end
        
        
    elseif strcmp(opt,'geom')
        
        geom=datainput{kf};
        
        if ~isempty(geom)
            
            node=geom.Node;
            quad=geom.Quad;
            tria=geom.Tria;
            
            % store data
            [fem, nDom, nnodes, nquads, ntrias]=localStoreData(fem, nDom, node, quad, tria, nnodes, nquads, ntrias);   
        end
        
        
    end

end

fem.Sol.nDom=nDom-1;

% save out
varargout{1}=fem;

if nargout==2
   log.nNodes=nnodes;
   log.nQuads=nquads;
   log.nTrias=ntrias;
    
   varargout{2}=log;
end


% save component settings
function [fem, nDom, nnodes, nquads, ntrias]=localStoreData(fem, nDom, node, quad, tria, nnodes, nquads, ntrias)

% NODES
nnode=0;
if node(1)~=-1
    nnode=size(node,1); % actual
end

% QUAD  
nquad=0;
if quad(1)~=-1
    nquad=size(quad,1);

    quad=quad+nnodes;
end

% TRIA 
ntria=0;
if tria(1)~=-1
    ntria=size(tria,1);

    tria=tria+nnodes;
end        

fem.Domain(nDom).Status=true; % by default all elements of this domain are active

% connectivity
idelementquads=[1:nquad] + (nquads+ntrias);
idelementtrias=[1:ntria] + (nquads+ntrias + nquad);

fem.Domain(nDom).Element = [1:nquad+ntria] + (nquads+ntrias);
fem.Domain(nDom).ElementQuad = idelementquads;
fem.Domain(nDom).ElementTria = idelementtrias;

idnodes = [1:nnode] + nnodes;

fem.Domain(nDom).Node = idnodes;

% material and constants (initial values)
fem.Domain(nDom).Material.E=210e3;
fem.Domain(nDom).Material.ni=0.33; 
fem.Domain(nDom).Material.lamda=5/6; 
fem.Domain(nDom).Material.Density=1; 

fem.Domain(nDom).Constant.Th=1; 

fem.Domain(nDom).NormalFlip=false;

% domain load
fem.Domain(nDom).Load.Value=[0 0 0 0 0 0]; % zero by default
fem.Domain(nDom).Load.Flag=false; % false by default
fem.Domain(nDom).Load.Frame='ref';

% Sub-modelling options
%--------------------------------------------------------------------------

fem.Domain(nDom).SubModel.CuttingSt=[1 2 3]; 
fem.Domain(nDom).SubModel.CuttingId=[1 2 3]; 
fem.Domain(nDom).SubModel.SearchDist=10; 

% save mesh
fem=storeNodes(fem, node, idnodes);

fem=storeQuads(fem, quad, nDom, idelementquads);
fem=storeTrias(fem, tria, nDom, idelementtrias);

% update counters 
nnodes=nnodes+nnode;
nquads=nquads+nquad;
ntrias=ntrias+ntria;

nDom=nDom+1;

