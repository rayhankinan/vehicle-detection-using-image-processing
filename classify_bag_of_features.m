function labels = classify_bag_of_features(categoryClassifier, I)

% Classify the image
[labelIdx, ~] = predict(categoryClassifier, I);

% Find the label with the highest score
labels = categoryClassifier.Labels(labelIdx);

end