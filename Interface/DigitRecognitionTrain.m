
function DigitRecognitionTrain()
[trainImages,trainLabels,testImages,testLabels]=DigitRecognitionDataset();
layers=DigitRecognitionModel();

options = trainingOptions('adam',...
'Plots','training-progress',...
'LearnRateSchedule','piecewise',...
'LearnRateDropFactor',0.2,...
'LearnRateDropPeriod',5,...
'MaxEpochs',50,...
'MiniBatchSize',256);

disp(options);

trainNet = trainNetwork(trainImages,trainLabels,layers,options);
save MNIST_LeNet5 trainNet
testOutput = classify(trainNet, testImages);
accuracy = sum(testOutput==testLabels)/numel(testLabels);
fprintf("The accuracy on test dataset is %f.\n", accuracy);
end