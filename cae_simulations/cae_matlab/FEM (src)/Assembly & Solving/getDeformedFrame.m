% get deformed frame
function fem=getDeformedFrame(fem)
 
 % get deformed frame
 fem=getDeformedFrameMatrix(fem);
 
 % update node normals
 fem=getNodeNormalDeformedFrame(fem);
     
 
 