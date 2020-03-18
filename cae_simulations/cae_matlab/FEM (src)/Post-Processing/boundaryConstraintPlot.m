% plot boundary constraints
function boundaryConstraintPlot(fem)

% bilateral
if fem.Post.ShowBoundary.BilateralNode
    bilateralNodeBcPlot(fem);
end

if fem.Post.ShowBoundary.BilateralElement
    bilateralElementBcPlot(fem);
end

% unilateral
if fem.Post.ShowBoundary.Unilateral
    unilateralBcPlot(fem);
end

% pin-hole
if fem.Post.ShowBoundary.PinHole
    pinholeBcPlot(fem);
end

% pin-slot
if fem.Post.ShowBoundary.PinSlot
    pinslotBcPlot(fem);
end

% dimple 
if fem.Post.ShowBoundary.Dimple
    dimpleBcPlot(fem);
end

% rigid link
if fem.Post.ShowBoundary.RigidLink
    rigidlinkBcPlot(fem);
end

% contact pair
if fem.Post.ShowBoundary.Contact
    contactPairPlot(fem);
end


