getImages(10,'./TestImages');

function getImages(num, storePath)
[fileId, errmsg] = fopen('Mnist/t10k-images-idx3-ubyte', 'r', 'b');
if fileId == -1
    error(errmsg);
end
magicNumber = fread(fileId, 1, 'int32', 0, 'b');
if magicNumber == 2051
    fprintf("Valid images data! Writing the images...\n");
end
imagesNumber = fread(fileId, 1, 'int32', 0, 'b');
rowNumber = fread(fileId, 1, 'int32', 0, 'b');
colNumber = fread(fileId, 1, 'int32', 0, 'b');
fprintf("Number of images in the dataset: %d.\n", imagesNumber);
data = fread(fileId, inf, 'unsigned char');
data = reshape(data, colNumber, rowNumber, imagesNumber);
% 轴转置, 变化为[H W N]
data = permute(data, [2 1 3]);

if storePath(end:end)~='/'
    storePath(end+1)='/';
end

for i=1:num
    imwrite(data(:,:,i),strcat(storePath,strcat(num2str(i),".jpg")));
end

fclose(fileId);
end