function layers = CurveFittingModel(dimension, layer_num, activeFunc, neurons)
% 创建模型
lgraph = layerGraph();
templayers = featureInputLayer(dimension-1,"Name","input");
lgraph = addLayers(lgraph,templayers);
for i=1:layer_num
    templayers = fullyConnectedLayer(neurons,"Name", strcat("fc",num2str(i)));
    lgraph = addLayers(lgraph,templayers);
    switch activeFunc
        case "Relu"
            templayers = reluLayer("Name",strcat("relu",num2str(i)));
        case "LeakyReulu"
            templayers = leakyReluLayer(0.01,"Name",strcat("leakyRelu",num2str(i)));
        case "ClippedRelu"
            templayers = clippedReluLayer(3,"Name",strcat("clippedrelu",num2str(i)));
        case "Tanh"
            templayers = tanhLayer("Name",strcat("tanh",num2str(i)));
        case "Swish"
            templayers = swishLayer("Name",strcat("swish",num2str(i)));
        case "Elu"
            templayers = eluLayer(1,"Name",strcat("elu",num2str(i)));
    end
    lgraph = addLayers(lgraph,templayers);
end
templayers = fullyConnectedLayer(1,"Name","out");
lgraph = addLayers(lgraph,templayers);
templayers = regressionLayer("Name","regressionoutput");
lgraph = addLayers(lgraph,templayers);
layers = lgraph.Layers;
end