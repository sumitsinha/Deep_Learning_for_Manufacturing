function model=cross_member_part_contact(model,contactfile)
model.database=modelImportInput(model.database, contactfile, 'Contact');
for i=1:length(model.database.Input.Contact)
    model.database.Input.Contact(i).SearchDist=[20 10]; % normal distance/sharp angle
    model.database.Input.Contact(i).Sampling=0.2;
end
model.database.Input.Contact(3).MasterFlip=true;
model.database.Input.Contact(3).Offset=1.8;
% %
end