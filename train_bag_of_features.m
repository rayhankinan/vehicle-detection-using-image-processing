function train_bag_of_features()

% Load the image collection using an imageDatastore
imds = imageDatastore('./Dataset','IncludeSubfolders',true,'LabelSource','foldernames');

% Separate the sets into training and validation data
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomize');

% Extracts SIFT features from all images in all image categories
% Constructs the visual vocabulary by reducing the number of features through quantization of feature space using K-means clustering
bag = bagOfFeatures(trainingSet, "CustomExtractor", @custom_extractor, "TreeProperties", [1 1000]);

% Encoded training images from each category are fed into a SVM classifier
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);

% Evaluate the classifier on the validationSet
evaluate(categoryClassifier, validationSet);

% Save the classifier
save('./Model/categoryClassifier.mat', 'categoryClassifier');

end