function model=cross_member_part_property (model,mhfile)

%% Define material properties
%% Define material properties
% * Thickness
% * Young's Modulus
% * Poisson's ratio
Th=[2.02 1.95 1.8 1.95];
E=[210e6 210e6 210e6 210e6];
nu=[0.303 0.303 0.303 0.303];
%
nmesh=length(mhfile);
col={'r','c','b','g'};
for i=1:nmesh
    model.database=modelAddItem(model.database, 'Part');
    model.database.Input.Part(i).E=E(i);
    model.database.Input.Part(i).nu=nu(i);
    model.database.Input.Part(i).Th=Th(i);
    model.database.Input.Part(i).Mesh{1}=mhfile{i};
    %
    model.database.Input.Part(i).Graphic.ShowEdge=false;
    model.database.Input.Part(i).Graphic.FaceAlpha=0.6;
    model.database.Input.Part(i).Graphic.ShowNormal=false;
    model.database.Input.Part(i).Graphic.Color=col{i};
end

end