function train_alexnet()
    % Load pretrained alexnet
    net = alexnet;

    % Load data and split data for training and validation 
    imds = imageDatastore("../Dataset/", "IncludeSubfolders", true, "LabelSource", "foldernames");
    [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

    % Get input size
    inputSize = net.Layers(1).InputSize;
    % Get convolutional + relu + pooling layers
    layersTransfer = net.Layers(1:end-3);
    % Get number of classes in dataset
    numClasses = numel(categories(imdsTrain.Labels));
    % Construct layers
    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer
    ];

    % Augmentation
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, 'ColorPreprocessing', 'gray2rgb');

    % Train with mini-batch gradient descent
    options = trainingOptions('sgdm', 'MiniBatchSize', 10, 'MaxEpochs', 10, 'InitialLearnRate', 1e-4, 'Shuffle', 'every-epoch', 'ValidationData', augimdsValidation, 'ValidationFrequency', 3, 'Verbose', false, 'Plots', 'training-progress');
    net = trainNetwork(augimdsTrain, layers, options);

    % Save model
    save("alexnet.mat", "net");
end