function model=cross_member_clampM (model,clampMfile)

model.database=modelImportInput(model.database, clampMfile, 'ClampM');
for i=1:length(model.database.Input.Locator.ClampM)
    model.database.Input.Locator.ClampM(i).SearchDist=[50 10];
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.L=20;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.D=10;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.A=10;
    model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.B=10;
    
    if i<=2
        model.database.Input.Locator.ClampM(i).Geometry.Shape.Type{1}=2;
    end
    
    if i>=21 && i<=27
        model.database.Input.Locator.ClampM(i).FlipNormal=true;
    end
end


% function model=cross_member_clampM (model,clampMfile)
% 
% model.database=modelImportInput(model.database, clampMfile, 'ClampM');
% for i=1:length(model.database.Input.Locator.ClampM)
%     model.database.Input.Locator.ClampM(i).SearchDist=[50 10];
%     model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.L=20;
%     model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.D=10;
%     model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.A=10;
%     model.database.Input.Locator.ClampM(i).Geometry.Shape.Parameter.B=10;
%     
%     if i<=2
%         model.database.Input.Locator.ClampM(i).Geometry.Shape.Type{1}=2;
%     end
%     % Manually Change Nt to move clamp in Flange direction
%     % Clamp 1
%     if i==1
%         model.database.Input.Locator.ClampM(i).TangentType{1}=1;
%         model.database.Input.Locator.ClampM(i).Nt=[0.8 0.3 -0.4];
%         model.database.Input.Locator.ClampM(i).NtReset=[0.8 0.3 -0.4];
%     end
%     % Clamp 2
%     if i==2
%         model.database.Input.Locator.ClampM(i).TangentType{1}=1;
%         model.database.Input.Locator.ClampM(i).Nt=[0.8 0.2 -0.4];
%         model.database.Input.Locator.ClampM(i).NtReset=[0.8 0.2 -0.4];
%     end
% end