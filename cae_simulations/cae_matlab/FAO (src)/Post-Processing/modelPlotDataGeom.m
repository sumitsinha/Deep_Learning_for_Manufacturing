% plot products
function modelPlotDataGeom(data, field, tag)

if nargin==2
    tag='';
    logPanel.Panel=[];
    logPanel.motionData=[];
end

if nargin==3
    logPanel.Panel=[];
    logPanel.motionData=[];
end
            
% plot parts
npart=0;
if strcmp(field, 'Part') && isfield(data.database.Input, 'Part')
    npart=length(data.database.Input.Part);
elseif strcmp(field, 'PartG') && isfield(data.database.Input, 'PartGraphic')
    npart=length(data.database.Input.PartGraphic);
end

if npart==0
    return
end

if strcmp(field, 'Part') 
    data.database.Model.Nominal.Post.Options.ParentAxes=data.Axes3D.Axes;
    data.database.Model.Nominal.Post.Options.ShowAxes=data.Axes3D.Options.ShowAxes;
    data.database.Model.Nominal.Post.Options.SymbolSize=data.Axes3D.Options.SymbolSize;
    data.database.Model.Nominal.Post.Options.LengthAxis=data.Axes3D.Options.LengthAxis;
    data.database.Model.Nominal.Post.Options.SubSampling=data.Axes3D.Options.SubSampling;
 elseif strcmp(field, 'PartG')
    data.database.Model.Graphic.Post.Options.ParentAxes=data.Axes3D.Axes;
    data.database.Model.Graphic.Post.Options.ShowAxes=data.Axes3D.Options.ShowAxes;
    data.database.Model.Graphic.Post.Options.SymbolSize=data.Axes3D.Options.SymbolSize;    
end

% plot parts
if strcmp(field, 'Part') 
    npart=length(data.database.Input.Part);
elseif strcmp(field, 'PartG')
     npart=length(data.database.Input.PartGraphic);
end

for i=1:npart
    
    if strcmp(field, 'Part') 
        st=data.database.Input.Part(i).Status;
    elseif strcmp(field, 'PartG')
        st=data.database.Input.PartGraphic(i).Status;
    end
    
    if st==0 && data.database.Input.Part(i).Enable
        if strcmp(field, 'Part') 
            modelPlotDataGeomSingle(data.database, field, i, [], tag, logPanel)
        elseif strcmp(field, 'PartG')
            modelPlotDataGeomSingle(data.database, field, i, data.database.Input.PartGraphic(i).Domain, tag, logPanel)
        end
    end 
end
