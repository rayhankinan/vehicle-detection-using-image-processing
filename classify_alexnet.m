function labels = classify_alexnet(I)

% Load the pretrained AlexNet network
load("./Model/alexnet.mat", "net");

% Classify the image using AlexNet
labels = classify(net, I);

end