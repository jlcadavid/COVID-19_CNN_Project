clear all;
close all;
clc;

%% Load and Explore Image Data

COVID19DatasetPath = fullfile('/', 'home', 'jlcadavid', 'Desktop', 'COVID-19 Project', 'CODE19 Data');
imds = imageDatastore(COVID19DatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

imageSize = [50 60 3];

imds.ReadFcn = @customReadDatastoreImage;

% Number of Images
num_images=length(imds.Labels);

% Visualize random images
perm=randperm(num_images,6);
figure;
for idx=1:length(perm)
    
    subplot(2,3,idx);
    imshow(imread(imds.Files{perm(idx)}));
    title(sprintf('%s',imds.Labels(perm(idx))))
    
end

%% Specify Training and Validation Sets

numTrainFiles = round(min(labelCount.Count)*0.7)
[imdsTrain, imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3]);

%augimds = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', imageAugmenter)

%% Define Convolutional Neural Network Architecture

layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(7,20,'Stride', 1, 'Padding','same')
    batchNormalizationLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(7,50,'Stride', 1,'Padding','same')
    batchNormalizationLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(7,500,'Stride', 1,'Padding','same')
    batchNormalizationLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    reluLayer
    
    convolution2dLayer(1, 2,'Stride', 1,'Padding','same')
    batchNormalizationLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    
    classificationLayer];

%% Specify Training Options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train Network Using Training Data

net = trainNetwork(imdsTrain,layers,options);

%% Classify Validation Images and Compute Accuracy

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%% Functions

function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[50 60]);
end