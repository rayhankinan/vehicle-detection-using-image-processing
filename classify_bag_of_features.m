function labels = classify_bag_of_features(I)

% Load the classifier
load("./Model/categoryClassifier.mat", "categoryClassifier")

% Classify the image
[labelIdx, ~] = predict(categoryClassifier, I);

% Find the label with the highest score
labels = categoryClassifier.Labels(labelIdx);


end