function model=cross_member_clampS (model,clampSfile)

model.database=modelImportInput(model.database, clampSfile, 'ClampS');
%model.database=modelImportInput(model.database, clampSfile, 'ClampS',{'Nt'});
for i=1:length(model.database.Input.Locator.ClampS)
    model.database.Input.Locator.ClampS(i).SearchDist=[50 10];
    model.database.Input.Locator.ClampS(i).NormalType{1}=1;
    model.database.Input.Locator.ClampS(i).TangentType{1}=1;
    %
    model.database.Input.Locator.ClampS(i).Nt=[1 0 0];
    model.database.Input.Locator.ClampS(i).NtReset=[1 0 0];
    if i<=2 || i>=6
        model.database.Input.Locator.ClampS(i).Nm=[0 1 0];
        model.database.Input.Locator.ClampS(i).NmReset=[0 1 0];
    else
        model.database.Input.Locator.ClampS(i).Nm=[0 0 1];
        model.database.Input.Locator.ClampS(i).NmReset=[0 0 1];
    end
    
    % Manually Change Nt to move clamp in Flange direction
    % Clamp 1
    if i==1
        model.database.Input.Locator.ClampS(i).Nt=[1 0 1];
        model.database.Input.Locator.ClampS(i).NtReset=[1 0 1];
    end
    % Clamp 9
    if i==9
        model.database.Input.Locator.ClampS(i).Nt=[0.9 0 -0.4];
        model.database.Input.Locator.ClampS(i).NtReset=[0.9 0 -0.4];
    end
    
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Type{1}=2;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.L=20;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.D=15;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.A=15;
    model.database.Input.Locator.ClampS(i).Geometry.Shape.Parameter.B=15;
    model.database.Input.Locator.ClampS(i).Graphic.Color='g';
end
end