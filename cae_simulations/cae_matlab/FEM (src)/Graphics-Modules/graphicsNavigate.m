% navigate throught the 3D renderer (pan, zoom, rotate)
function graphicsNavigate(fem, redfc)

% INPUT:
 % fem: fem structure
 % redfc: downsampling factor

if nargin==0
    ax=gca;
    fig=gcf;
else
    ax=fem.Post.Options.ParentAxes;
    fig=get(ax,'parent');
end

visopt=get(ax, 'visible');

%--
f=uimenu(fig, 'Label','Navigate TOOL');
uimenu(f,'Label','Rotate 3D','Callback',{@setUpGraphicsNav, fig, ax, visopt, redfc, 'rotate'});
uimenu(f,'Label','Zoom','Callback',{@setUpGraphicsNav, fig, ax, visopt, redfc, 'zoom'});
uimenu(f,'Label','Pan','Callback',{@setUpGraphicsNav, fig, ax, visopt, redfc, 'pan'});


