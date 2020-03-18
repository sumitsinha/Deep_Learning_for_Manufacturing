 % export 3D PDF file
function export3dPdfFile(h, filename)

% h: axes handle
% filename: filename

[~, filepdf, ~] = fileparts(filename);

% delete previuos file
delete(filename);

% save new file
fig2pdf3d(h, filepdf, 'movie15', 'pdflatex')

