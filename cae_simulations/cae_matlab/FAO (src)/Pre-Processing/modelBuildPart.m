% build part geometry
function data=modelBuildPart(data, opt)
    
% data: input data model
% opt: [logical]
    % opt(1): true/false => true: import all geometry
    % opt(2): true/false => true: compute part UCS
    % opt(3): true/false => true: apply placement
    % opt(4): true/false => true: compute stiffness matrix 
    % opt(5): true/false => true: reset model (both part and assembly variables)

% Set default option
if nargin==1
    opt=true(1,5);
else
    if length(opt)~=5
        error('Model build (part): wrong input format');
    end
end

%--
if ~isfield(data.Input,'Part')
    return
end
%
data.Model.Nominal.Options.StiffnessUpdate=false; % do not udpate stiffness matrix
data.Model.Nominal.Options.ConnectivityUpdate=false; % DO NOT build connectivity matrix
data.Model.Nominal.Options.UseActiveSelection=false; % DO NOT use active selection
%-------------------------

%--
if opt(1)
    
    %--
    % STEP 1: read file names
    npart=length(data.Input.Part);

    filemesh=cell(1,npart);
    for i=1:npart
        file=data.Input.Part(i).Mesh{1}; % in case of multiple files only the first one is used

        if isempty(file)
                error('Building model: mesh file @ part[%g] not recognised',i);
        end

        filemesh{i}=file;

    end

    % STEP 2: import mesh
    try
        data.Model.Nominal=importMultiMesh(data.Model.Nominal, filemesh, 'file');
    catch
        error('Building model: failed to import parts');
    end
    %----------------

end

% STEP 3: update model
nd=data.Model.Nominal.Sol.nDom;

activeNode=[];
for i=1:nd

    % set material properties
    data.Model.Nominal.Domain(i).Material.E=data.Input.Part(i).E;
    data.Model.Nominal.Domain(i).Material.ni=data.Input.Part(i).nu;
    data.Model.Nominal.Domain(i).Constant.Th=data.Input.Part(i).Th;

    sta=data.Input.Part(i).Enable;

    % disable not active domains
    if ~sta % part NOT active
        data.Model.Nominal.Domain(i).Status=false;
        data.Input.Part(i).Status=-1; 
        
        activeNode(i).Status = false; % not active
    else
        data.Model.Nominal.Domain(i).Status=true;
        data.Input.Part(i).Status=0; 
        
        activeNode(i).Status = true; % not active
    end
    
    %--
    data.Input.Part(i).Selection.Node=data.Model.Nominal.Domain(i).Node;
    
    %-- 
    activeNode(i).Node = data.Input.Part(i).Selection.Node; %#ok<*AGROW>
    activeNode(i).Part=i; 
    
    %---------------
    % read placement matrix and update part's placement...
    Tucs=data.Input.Part(i).Placement.UCS;
    if opt(2)
        Tucs=getPartUCS(data, i);
    end
    data.Input.Part(i).Placement.UCS=Tucs;
    data.Input.Part(i).Placement.UCSreset=Tucs;
    T0w=data.Input.Part(i).Placement.T;
    
    %---------------
    % apply placement
    if opt(3)
        idnodes=data.Model.Nominal.Domain(i).Node;
        data.Model.Nominal.xMesh.Node.Coordinate(idnodes,:)=...
                apply4x4(data.Model.Nominal.xMesh.Node.CoordinateReset(idnodes,:),...
                         T0w(1:3,1:3), T0w(1:3,4)');
    end
    
    % ... and, now set status==0 (computed)
    if sta
        data.Input.Part(i).Status=0;
    else
        data.Input.Part(i).Status=-1;
    end
end

%- pre-process model
if nd>0
    if opt(4)
        data.Model.Nominal.Options.StiffnessUpdate=true; 
    else
        data.Model.Nominal.Options.StiffnessUpdate=false; 
    end
    data.Model.Nominal=femPreProcessing(data.Model.Nominal, activeNode);
end

% check normal flipping
for i=1:nd
    if data.Input.Part(i).FlipNormal % then flip normal
        data.Model.Nominal=flipNormalComponent(data.Model.Nominal, i);
    end
end

% Reset solution
if opt(5)
    data=modelReset(data, true, true); 
end

