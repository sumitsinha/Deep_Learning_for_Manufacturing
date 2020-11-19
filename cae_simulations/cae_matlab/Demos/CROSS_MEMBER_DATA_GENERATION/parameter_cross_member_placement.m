
function model=parameter_cross_member_placement (model)

% Positoning Station 1 (stage 2), Part 1
model.database.Input.Part(1).Parametrisation.Type(2,3)=1; % Model training paramater z Rot
model.database.Input.Part(1).Parametrisation.Type(2,4)=1; % Model training paramater x dev
model.database.Input.Part(1).Parametrisation.Type(2,5)=1; % Model training paramater y dev
model.database.Input.Part(1).Parametrisation.UCS(2)=1; % use local UCS

% Positoning Station 3, Subassembly 1 (Part 1 (master)+ Part 2 (slave))
model.database.Input.Part(1).Parametrisation.Type(10,3)=1; % Model training paramater z Rot
model.database.Input.Part(1).Parametrisation.Type(10,4)=1; % Model training paramater x dev
model.database.Input.Part(1).Parametrisation.Type(10,5)=1; % Model training paramater y dev
model.database.Input.Part(1).Parametrisation.UCS(10)=1; % use local UCS

% Positoning Station 2 (stage 6), Part 3
model.database.Input.Part(3).Parametrisation.Type(6,3)=1; % Model training paramater z Rot
model.database.Input.Part(3).Parametrisation.Type(6,4)=1; % Model training paramater x dev
model.database.Input.Part(3).Parametrisation.Type(6,5)=1; % Model training paramater y dev
model.database.Input.Part(3).Parametrisation.UCS(6)=1; % use local UCS

end