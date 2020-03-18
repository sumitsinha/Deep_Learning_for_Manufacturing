% write stl file based on data structure
function exportStlFileTemplate(filename,...
                               datat)
% data.Vertex
% data.Quad
% data.Tria

% open file
idf=fopen(filename,'w');

for kobj=1:length(datat)
    
    nameObj=['solid_obj_', num2str(kobj)];
    fprintf(idf,'solid %s\r\n',nameObj);  

    data=datat(kobj);
    
    nele=length(data.Tria);
    for k=1:nele

        facek=data.Tria(k,:);
        vertex=data.Vertex(facek,:);
        normal=faceNormal(vertex(1,:),vertex(2,:),vertex(3,:));

        % face 1
        fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', normal(1),normal(2),normal(3) );
        fprintf(idf,'outer loop\r\n');        
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(2,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
        fprintf(idf,'endloop\r\n');
        fprintf(idf,'endfacet\r\n');

    end

    nele=length(data.Quad);
    for k=1:nele

        facek=data.Quad(k,:);

        vertex=data.Vertex(facek,:);

        % face 1
        normal=faceNormal(vertex(1,:),vertex(2,:),vertex(3,:));

        fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', normal(1),normal(2),normal(3) );
        fprintf(idf,'outer loop\r\n');        
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(2,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
        fprintf(idf,'endloop\r\n');
        fprintf(idf,'endfacet\r\n');

        % face 2
        normal=faceNormal(vertex(3,:),vertex(4,:),vertex(1,:));

        fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', normal(1),normal(2),normal(3) );
        fprintf(idf,'outer loop\r\n');        
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(4,:));
        fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
        fprintf(idf,'endloop\r\n');
        fprintf(idf,'endfacet\r\n');

    end


    fprintf(idf,'endsolid %s\r\n',nameObj);  

end

fclose(idf); % close file...


%--
function normal=faceNormal(P1,P2,P3)

v1 = P2-P1;
v2 = P3-P1;

v3 = cross(v1,v2);

normal = v3/norm(v3);


