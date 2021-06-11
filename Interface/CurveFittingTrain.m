function fitY = CurveFittingTrain(net,epoch,batchSize,X,Y)
options = trainingOptions('adam',...
'Plots','training-progress',...
'MaxEpochs',epoch,...
'MiniBatchSize',batchSize);
CurveFittingNet = trainNetwork(X',Y',net,options);
save CurveFittingNet CurveFittingNet

fitY = predict(CurveFittingNet, X');
end