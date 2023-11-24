function [features, featureMetrics] = custom_extractor(I)

% Convert image to grayscale if it is not already
if size(I, 3) == 3
    I = rgb2gray(I);
end

% Extract features using your custom method
% For example, using SURF features
points = detectSURFFeatures(I);
[features, valid_points] = extractFeatures(I, points);

% Define feature metrics (e.g., the strongest features)
% This is used to select the strongest features in bagOfFeatures
featureMetrics = valid_points.Metric;

end