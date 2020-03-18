% run robust polynomial fitting
function varargout=getRobustPolynomialModel(X, Y, degree,...
                                              numFolds,...
                                              numToLeaveOut,...
                                              maxIter,...
                                              ssize,...
                                              t,...
                                              p)

% INPUT:
% X: independent variables [nsample, nvars]
% Y: dependent variables [nsample, 1] 
% degree: [min max] polynomial degree
% numFolds: no. of cross validations
% numToLeaveOut: no. of point to leave out
% maxIter: maximum ransac refiniments
% ssize: minimum number of samples
% t:  The distance threshold between a data point and the model used to decide whether the point is an inlier or not
% p: desired probability of choosing at least one sample free from outliers

% OUTPUT
% varargout{1}: model after cleaning
% varargout{2}: model before cleaning
% varargout{3}: inliers poins ids

% STEP 1: evaluate best model
polymodelb=getPolynomialModel(X, Y, degree,...
                                      numFolds,...
                                      numToLeaveOut);

% STEP 2: remove outliers 
inliers=getPolyInliersRansac(X, Y, polymodelb.Degree,...
                                maxIter,...
                                ssize,...
                                t,...
                                p);
 
% STEP 3: fit the model on the cleaned data
degree=[polymodelb.Degree polymodelb.Degree];
polymodela=getPolynomialModel(X(inliers,:), Y(inliers), degree,...
                                      numFolds,...
                                      numToLeaveOut);

% save outputs
varargout{1}=polymodela;

if nargout==2
    varargout{2}=polymodelb;
end

if nargout==3
    varargout{2}=polymodelb;
    varargout{3}=inliers;
end

