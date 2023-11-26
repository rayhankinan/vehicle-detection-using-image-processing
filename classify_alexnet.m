function labels = classify_alexnet(net, I)

% Resize the image to the input size of the network
I = imresize(I, net.Layers(1).InputSize(1:2));

% Classify the image using AlexNet
labels = classify(net, I);

end