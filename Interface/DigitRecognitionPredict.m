function [pre,score]=DigitRecognitionPredict(srcImage, showProcessing)
close all;

if showProcessing
    figure('name', '前期处理');
    subplot(221), imshow(srcImage), title('原始图像');
end

srcImage = imcomplement(srcImage);

% 采用otsu方法获取二值化阈值，进行二值化并进行显示
threshold = graythresh(srcImage); %使用最大类间方差法找到图片的一个合适的阈值（threshold）
srcImage = im2gray(srcImage);
bwImage = imbinarize(srcImage, threshold); 
if showProcessing
    subplot(222), imshow(bwImage), title('二值图像');
end


se = strel('disk', 2); % 创建一个半径为2的圆形结构元
open_bwImage = imopen(bwImage, se); % 开操作是先腐蚀再膨胀

if showProcessing
    subplot(223), imshow(open_bwImage), title('开操作图像'); 
end

L = bwlabel(open_bwImage, 4); % 二维二进制图像中的连通域标记。n=4,说明按照4连通域寻找目标
RGB = label2rgb(L, 'spring', 'c', 'shuffle');

if showProcessing
    subplot(224), imshow(RGB), title('彩色图');
end

stats = regionprops(open_bwImage,'basic');
Ar = cat(1, stats.Area); 
ind = find(Ar == max(Ar));
open_bwImage = ismember(L, ind);

if showProcessing
    figure('name', '标记提取');
    subplot(221), imshow(open_bwImage), title('仅保留最大连通域');
end

stats = regionprops(open_bwImage,'basic'); 

if showProcessing
    subplot(222), imshow(open_bwImage), title('质心'); 
    hold on;
    
% 标出质心
plot(stats.Centroid(:, 1), stats.Centroid(:, 2), 'r*'); 
 
% 绘制感兴趣区域ROI
rectangle('position', [stats.BoundingBox], ...
           'LineWidth', 2, 'LineStyle', '-', 'EdgeColor', 'r');
hold off; 

end


crop_srcImage = imcrop(srcImage, [stats.BoundingBox]);
if showProcessing
    subplot(223), imshow(crop_srcImage), title('提取之后的原始图像'); 
end
 
% 对二值图像的提取
bounding = stats.BoundingBox;
width = bounding(3);
height = bounding(4);
len = max(width,height);
center = [stats.Centroid(:, 1) stats.Centroid(:, 2)];

bounding(1) = center(1) - len/2;
bounding(2) = center(2) - len/2;
bounding(3) = len;
bounding(4) = len;

crop_open_bwImage = imcrop(open_bwImage, bounding); 
if showProcessing
    subplot(224), imshow(crop_open_bwImage), title('提取之后的二值图像');
end

test_image = double(crop_open_bwImage);
test_image = imresize(test_image, [28,28]); %保证输入为28*28
threshold = graythresh(test_image); 

test_image = imbinarize(test_image,threshold); %二值化

if showProcessing
    figure;imshow(test_image);title("输入网络的图像");
end

load("Model/MNIST_LeNet5");
[Predict,Score] = classify(trainNet, test_image);
Predict = string(Predict);
Predict = double(Predict);
fprintf("The prediction is %d, the probability is %f.\n",Predict,Score(Predict+1));
pre = Predict;
score = Score(Predict+1)*100;
end