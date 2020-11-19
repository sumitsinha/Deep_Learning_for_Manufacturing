% plot single product
function modelPlotDataGeomSingle(data, field, id, iddom, tag, logPanel)

if nargin<4
    iddom=[];
    tag=sprintf('part[%g]', id);
    logPanel.Panel=[];
    logPanel.motionData=[];
end

if nargin<5
    tag=sprintf('part[%g]', id);
    logPanel.Panel=[];
    logPanel.motionData=[];
end

if nargin<6
    logPanel.Panel=[];
    logPanel.motionData=[];
end

% plot
if strcmp(field,'Part')
    f=data.Model.Nominal;

    % set options
    f.Post.Options.ColorPatch=data.Input.Part(id).Graphic.Color;
    f.Post.Options.FaceAlpha=data.Input.Part(id).Graphic.FaceAlpha;
    f.Post.Options.ShowEdge=data.Input.Part(id).Graphic.ShowEdge;
    f.Post.Options.ShowPath=data.Input.Part(id).Graphic.Show;
    
    if data.Input.Part(id).Graphic.Show
        meshComponentPlot(f, id, tag, logPanel)
    end

    % show normal
    if data.Input.Part(id).Graphic.ShowNormal 
        normalElementPlot(f, id, tag);
    end
    
    % Show UCS
    if data.Input.Part(id).Placement.ShowFrame
        T=data.Input.Part(id).Placement.UCS;
        plotFrame(T(1:3,1:3), T(1:3,4)',...
                  data.Model.Nominal.Post.Options.ParentAxes,...
                  data.Model.Nominal.Post.Options.LengthAxis,...
                  tag);
    end
    
elseif strcmp(field,'PartG')
    f=data.Model.Graphic;

    % set options
    for i=1:length(iddom)
        f.Post.Options.ColorPatch=data.Input.PartGraphic(id).Graphic.Color(i, :);
        f.Post.Options.FaceAlpha=data.Input.PartGraphic(id).Graphic.FaceAlpha;
        f.Post.Options.ShowEdge=data.Input.PartGraphic(id).Graphic.ShowEdge;
        f.Post.Options.ShowPath=data.Input.PartGraphic(id).Graphic.Show;

        meshComponentPlot(f, iddom(i), tag)
    end
end
