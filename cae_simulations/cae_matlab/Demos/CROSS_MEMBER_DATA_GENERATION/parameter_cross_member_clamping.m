function model=parameter_cross_member_clamping (model)

%All ClampS have x,y,z deviation
for i=1:size(model.database.Input.Locator.ClampS,2)
    model.database.Input.Locator.ClampS(i).Parametrisation.Geometry.Type{1}{1}=8; % Model training paramater
    model.database.Input.Locator.ClampS(i).Parametrisation.Geometry.ShowFrame=true;
end

%Only Paramterizing First 2 ClampM rest parametrized in joining
for i=1:2
    model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.Type{1}{1}=8; % Model training paramater
    model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.ShowFrame=true;
end
end