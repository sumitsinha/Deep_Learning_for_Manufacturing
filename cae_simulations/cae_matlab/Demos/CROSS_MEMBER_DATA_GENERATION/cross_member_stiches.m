function model=cross_member_stiches (model,stitchfile)
model.database=modelImportInput(model.database, stitchfile, 'Stitch');
for i=1:length(model.database.Input.Stitch)
    model.database.Input.Stitch(i).Diameter=5;
    model.database.Input.Stitch(i).SearchDist=[50 10];
    model.database.Input.Stitch(i).Gap=100; % stitch is created if part-to-part <=.Gap
end
end