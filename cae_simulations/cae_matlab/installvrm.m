% Install VRM toolbox
function installvrm()
%
% Add to matlab the following paths:
    % (1) FEM (src)
    % (2) FAO (src)
    % (3) KINEMATICS (src)
    % (4) Demos
    % (5) GUI
    % ....
    % ....
%-----------------------------------------------------
% Copyright: Dr P. Franciosa, WMG/Uni. of Warwick 2020
% email: p.franciosa@warwick.ac.uk
% Tel: +44(0)2476573422 
%-----------------------------------------------------

%--
clc
clear all %#ok<CLALL>
close all
%
% check OS compatibility
fprintf('Installation of VRM started:\n');
if ismac % Max
    warning('Installation of VRM (warning): Mac not supported at the moment!')
    fprintf('   Installing VRM pack on Mac...\n');
elseif isunix % Linux
    warning('Installation of VRM (error): Linux not supported at the moment!')
    fprintf('   Installing VRM pack on Linux...\n');
elseif ispc % PC/win
    fprintf('   Installing VRM pack on Windows...\n');
else
    error('Installation of VRM (error): OS system not compatible!')
end
%
% install source codes
fprintf('     Installing FEM code...\n');
fem_src=fullfile(cd,'FEM (src)');
if ~exist(fem_src,'dir')
    warning('Installation of VRM (error): failed to locate "FEM(src)" folder!')
end
addpath(genpath(fem_src));
%
fprintf('     Installing FAO code...\n');
fao_src=fullfile(cd,'FAO (src)');
if ~exist(fao_src,'dir')
    warning('Installation of VRM (error): failed to locate "FAO(src)" folder!')
end
addpath(genpath(fao_src));
%
fprintf('     Installing Kinematics code...\n');
kine_src=fullfile(cd,'KINEMATICS (src)');
if ~exist(kine_src,'dir')
    warning('Installation of VRM (error): failed to locate "KINEMATICS (src)" folder!')
end
addpath(genpath(kine_src));
%
% install GUI
fprintf('     Installing GUI code...\n');
gui_src=fullfile(cd,'GUI');
if ~exist(gui_src,'dir')
    warning('Installation of VRM (error): failed to locate "GUI" folder!')
end
addpath(genpath(gui_src));
%
% install demo folder
fprintf('     Installing Demo pack...\n');
demos_src=fullfile(cd,'Demos');
if ~exist(demos_src,'dir')
    warning('Installation of VRM (error): failed to locate "Demos" folder!')
end
addpath(genpath(demos_src));
%
fprintf('Installation completed!\n');
%