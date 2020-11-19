function model=parameter_cross_member_joining (model)

% 25 (3-27) ClampM used as joints are paramterized using type 19 (Categorical + xyz)
for i=3:size(model.database.Input.Locator.ClampM,2)
    model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.Type{1}{1}=19; % Model training paramater
    model.database.Input.Locator.ClampM(i).Parametrisation.Geometry.ShowFrame=true;
end

% 25  stiches used joints are paramterized using type 19 (Categorical + xyz)
for i=1:size(model.database.Input.Stitch,2)
    model.database.Input.Stitch(i).Parametrisation.Geometry.Type{1}{1}=19; % Model training paramater
    model.database.Input.Stitch(i).Parametrisation.Geometry.ShowFrame=true;
end

end