%- Total Six part variation parameters
% 4 global paramters for each part (Twisting and Bending)
% 2 local paramters for part 3 and 4
function model=cross_member_part_variation (model, mmpfile, mmlfile)

%-- Define parameters for non-ideal part
nmfiles=length(mmpfile);
for k=1:nmfiles
    if ~isempty(mmpfile{k})
        model.database=modelImportInput(model.database, mmpfile{k}, 'Morphing'); 
        
        model.database.Input.Part(k).Geometry.Type{1}=2; % morphed
        model.database.Input.Part(k).Geometry.Mode{1}=1; % deterministic 
        correlation_length = importdata(mmlfile{k}); % this line should be made "user-safe"
        for i=1:length(model.database.Input.Part(k).Morphing)
            model.database.Input.Part(k).Morphing(i).Distribution{1}=1; % Model training paramater
            model.database.Input.Part(k).Morphing(i).NormalMode{1}=1; % user
            model.database.Input.Part(k).Morphing(i).Nc=[1 1 1];%Global PV is Y direction (Surface Normal), Local NC =[1 1 1] set manually below
            %--
            idSele=model.database.Input.Part(k).Morphing(i).Selection;
            model.database.Input.Selection(idSele).Rm=correlation_length(i,:); % Input correlation length for the selected Point
            model.database.Input.Selection(idSele).Type{1}=2; % Ellipsoide
        end
    end
% Manual setup For Local Deformation Patterns in Part 3 and 4 as Normal Variation    
model.database.Input.Part(3).Morphing(4).NormalMode{1}=2;
model.database.Input.Part(4).Morphing(5).NormalMode{1}=2;
%model.database.Input.Part(3).Morphing(4).Nc=[1 1 1];   
%model.database.Input.Part(4).Morphing(5).Nc=[1 1 1];   

end

% for i=1:3
%     model.database.Input.Part(3).Morphing(i).Selection=0;
% end

% for i=1:4
%     model.database.Input.Part(4).Morphing(i).Selection=0;
% end
