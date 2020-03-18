function point=createBezierPoint(Pc, n, t)

    % Pc: control points [n+1, 3]
    % n: curve degree
    % t: parameter list
    
    % point: bezier points

    nPoint=length(t);
    
    point=zeros(nPoint,3);
    
    % loop over all points
    for i=1:nPoint
        
        b=getBasicFunctionBezier(t(i),n);
        
        point(i,:)=b*Pc;
      
    end