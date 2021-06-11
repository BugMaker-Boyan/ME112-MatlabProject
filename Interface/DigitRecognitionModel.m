function layers=DigitRecognitionModel()
layers = [
    imageInputLayer([28 28 1],"Name","imageinput")
    convolution2dLayer([5 5],6,"Name","conv1","Padding","same")
    tanhLayer("Name","tanh1")
    maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same","Stride",[2 2])
    convolution2dLayer([5 5],16,"Name","conv2","Padding","same")
    tanhLayer("Name","tanh2")
    maxPooling2dLayer([2 2],"Name","maxpool","Padding","same","Stride",[2 2])
    fullyConnectedLayer(120,"Name","fc1")
    fullyConnectedLayer(84,"Name","fc2")
    fullyConnectedLayer(10,"Name","fc3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
end