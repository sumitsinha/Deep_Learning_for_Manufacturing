function [mhfile,mmpfile,mmlfile,pinholefile,pinslotfile,stitchfile,Ncblockfile,clampSfile,clampMfile,contactfile]=cross_member_input_files ()

%
%   Mesh file
%
mhfile{1}=fullfile(findfile(cd, '[1]Pocket.inp'),'[1]Pocket.inp');
mhfile{2}=fullfile(findfile(cd, '[2]Pocket_reinf.inp'),'[2]Pocket_reinf.inp');
mhfile{3}=fullfile(findfile(cd, '[3]Cross_member.inp'),'[3]Cross_member.inp');
mhfile{4}=fullfile(findfile(cd, '[4]Cross_member_reinf.inp'),'[4]Cross_member_reinf.inp');
%
%   Control points file for morphing mesh
%
% Importing Files with control Points 
mmpfile{1}=fullfile(findfile(cd, 'var_geom_mm_points_part1.txt'),'var_geom_mm_points_part1.txt');
mmpfile{2}=fullfile(findfile(cd, 'var_geom_mm_points_part2.txt'),'var_geom_mm_points_part2.txt');
mmpfile{3}=fullfile(findfile(cd, 'var_geom_mm_points_part3.txt'),'var_geom_mm_points_part3.txt');
mmpfile{4}=fullfile(findfile(cd, 'var_geom_mm_points_part4.txt'),'var_geom_mm_points_part4.txt');

% Importing Files with Correlation Length
mmlfile{1}=fullfile(findfile(cd, 'var_geom_mm_length_part1.txt'),'var_geom_mm_length_part1.txt');
mmlfile{2}=fullfile(findfile(cd, 'var_geom_mm_length_part2.txt'),'var_geom_mm_length_part2.txt');
mmlfile{3}=fullfile(findfile(cd, 'var_geom_mm_length_part3.txt'),'var_geom_mm_length_part3.txt');
mmlfile{4}=fullfile(findfile(cd, 'var_geom_mm_length_part4.txt'),'var_geom_mm_length_part4.txt');
% %
%   Hole file
%
pinholefile=fullfile(findfile(cd, 'hole_multi_station.txt'),'hole_multi_station.txt');
%
%   Slot file
%
pinslotfile=fullfile(findfile(cd, 'slot_multi_station.txt'),'slot_multi_station.txt');
%
%   Stitch file
%
stitchfile=fullfile(findfile(cd, 'stitch_multi_station.txt'),'stitch_multi_station.txt');
%
%   NCblock file
%
Ncblockfile=fullfile(findfile(cd, 'NCBlock_multi_station.txt'),'NCBlock_multi_station.txt');
%
%   ClampS file
%
clampSfile=fullfile(findfile(cd, 'clampS_multi_station.txt'),'clampS_multi_station.txt');
%
%   ClampM file
%
clampMfile=fullfile(findfile(cd, 'clampM_multi_station.txt'),'clampM_multi_station.txt');
%
%   Contact file
%
contactfile=fullfile(findfile(cd, 'contact_multi_station.txt'),'contact_multi_station.txt');
%
%

end