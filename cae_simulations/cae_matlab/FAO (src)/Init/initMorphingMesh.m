function d=initMorphingMesh()

d.Pc=[0 0 0]; % [X Y Z] (control point) 
d.Nc=[0 0 1]; % [Nx Ny Nz] (direction)
d.NormalMode={2,'User','Model'};
d.Parameter=[0 0.1]; % parameters of selected "Distribution". For Distribution=="Gaussian" => parameter=[mean, std]
d.Selection=1; % if "0" then use automatic selection; otherwise use "data.Input.Selection"
d.Distribution={1,'Deterministic', 'Gaussian'}; % distribution

%--
d.DeltaPc=0; % deviation of control point