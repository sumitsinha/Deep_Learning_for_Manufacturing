
function model=cross_member_placement (model, pinholefile, pinslotfile)

% Hole
model.database=modelImportInput(model.database, pinholefile, 'Hole');
for i=1:length(model.database.Input.PinLayout.Hole)
    model.database.Input.PinLayout.Hole(i).Geometry.Shape.Parameter.Diameter=20;
    model.database.Input.PinLayout.Hole(i).Parametrisation.Geometry.ShowFrame=true;
    model.database.Input.PinLayout.Hole(i).SearchDist=[10 10];
    model.database.Input.PinLayout.Hole(i).TangentType{1}=1;
    model.database.Input.PinLayout.Hole(i).Nt=[1 0 0];
    model.database.Input.PinLayout.Hole(i).NtReset=[1 0 0];
end
%
% Slot
model.database=modelImportInput(model.database, pinslotfile, 'Slot');
for i=1:length(model.database.Input.PinLayout.Slot)
    model.database.Input.PinLayout.Slot(i).Geometry.Shape.Parameter.Diameter=20;
    model.database.Input.PinLayout.Slot(i).Geometry.Shape.Parameter.Length=10;
    model.database.Input.PinLayout.Slot(i).Parametrisation.Geometry.ShowFrame=false;
    model.database.Input.PinLayout.Slot(i).SearchDist=[10 10];
end 

end