function model=cross_member_NCblock (model,Ncblockfile)
%% Define NCBlock
model.database=modelImportInput(model.database, Ncblockfile, 'NcBlock');
for i=1:length(model.database.Input.Locator.NcBlock)
    model.database.Input.Locator.NcBlock(i).SearchDist=[10 10];
    model.database.Input.Locator.NcBlock(i).NormalType{1}=1;
    model.database.Input.Locator.NcBlock(i).Nm=[0 1 0];
    model.database.Input.Locator.NcBlock(i).NmReset=[0 1 0];
    model.database.Input.Locator.NcBlock(i).Geometry.Shape.Type{1}=2;
    model.database.Input.Locator.NcBlock(i).Graphic.Color='g';
end
end