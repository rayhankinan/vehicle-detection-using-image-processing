function train_bag_of_features()

% Load the image collection using an imageDatastore
imds = imageDatastore('Dataset','IncludeSubfolders',true,'LabelSource','foldernames');

% Separate the sets into training and validation data
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');


% Extracts SURF features from all images in all image categories
% Constructs the visual vocabulary by reducing the number of features through quantization of feature space using K-means clustering
bag = bagOfFeatures(trainingSet);

% Encoded training images from each category are fed into a SVM classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% Evaluate the classifier on the validationSet
confMatrix = evaluate(categoryClassifier, validationSet);

% Display the confusion matrix
disp(confMatrix);

% Save the classifier
save('categoryClassifier.mat', 'categoryClassifier');

end