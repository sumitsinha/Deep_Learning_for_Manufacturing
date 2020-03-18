% plot labels
function annotationPlot(fem)

iddom=fem.Post.ShowAnnotation.Domain;

% node label
if fem.Post.ShowAnnotation.Node
    labelNodePlot(fem,iddom);
end

% element label
if fem.Post.ShowAnnotation.Element
    labelElementPlot(fem,iddom);
end

% normal node plot
if fem.Post.ShowAnnotation.NormalNode
    normalNodePlot(fem,iddom);
end

% normal element plot
if fem.Post.ShowAnnotation.NormalElement
    normalElementPlot(fem,iddom);
end

