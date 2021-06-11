function y = CurveFittingPredict(x)
load CurveFittingNet
y = predict(CurveFittingNet,x);
end

