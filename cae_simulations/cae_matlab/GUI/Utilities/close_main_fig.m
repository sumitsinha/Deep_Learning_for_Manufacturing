function close_main_fig(~, ~)

findfig=findobj('type','figure');
delete(findfig);

% % unload libraries
% wdir=cd;
% 
% rmpath(genpath([wdir,'\GUILayout-v1p17']));
% rmpath(genpath([wdir,'\Functions']));
% rmpath(genpath([wdir,'\VRM-code']));
% %------------------------------------ 
