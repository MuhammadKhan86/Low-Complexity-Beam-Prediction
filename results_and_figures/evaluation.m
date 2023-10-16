datasetSize = 25000;

simScenario = "1";
rng(100)
shuffledIdx = randperm(datasetSize);

load('EBS_CDL-D_DataSet.mat')
txBeamMatEBSD         = results.EBS.txBeamId(shuffledIdx,:);
rsrpMatEBSD           = results.EBS.rsrpMat(shuffledIdx,:);
[txBeamIdEBSD, ~]     = find(txBeamMatEBSD.');

load('EBS_CDL-E_DataSet.mat')
txBeamMatEBSE         = results.EBS.txBeamId(shuffledIdx,:);
rsrpMatEBSE           = results.EBS.rsrpMat(shuffledIdx,:);
[txBeamIdEBSE, ~]     = find(txBeamMatEBSE.');

load('HBS_CDL-D_DataSet.mat')
txBeamMatParentHBSD = results.HBS.txBeamIdParent(shuffledIdx,:);
[txBeamIdParentHBSD, ~] = find(txBeamMatParentHBSD.');
rsrpMatParentHBSD      = results.HBS.rsrpMatParent(shuffledIdx,:);

txBeamMatChildHBSD     = results.HBS.txBeamIdChild(shuffledIdx,:);
[txBeamIdChildHBSD, ~] = find(txBeamMatChildHBSD.');
rsrpMatChildHBSD       = results.HBS.rsrpMatChild(shuffledIdx,:);


load('HBS_CDL-E_DataSet.mat')
txBeamMatParentHBSE = results.HBS.txBeamIdParent(shuffledIdx,:);
[txBeamIdParentHBSE, ~] = find(txBeamMatParentHBSE.');
rsrpMatParentHBSE      = results.HBS.rsrpMatParent(shuffledIdx,:);

txBeamMatChildHBSE     = results.HBS.txBeamIdChild(shuffledIdx,:);
[txBeamIdChildHBSE, ~] = find(txBeamMatChildHBSE.');
rsrpMatChildHBSE       = results.HBS.rsrpMatChild(shuffledIdx,:);

if strcmpi(simScenario,'1')
  inputFeat     = rsrpMatParentHBSD;
  optLabels     = txBeamMatEBSD;
  optLabelsIdx  = txBeamIdEBSD;
  rsrpMatHBS    = rsrpMatChildHBSD;
  txBeamMatChildHBS = txBeamMatChildHBSD;
  txBeamMatParentHBS = txBeamMatParentHBSD;
  rsrpMatEBS = rsrpMatEBSD;
elseif strcmpi(simScenario,'2')
  inputFeat     = [rsrpMatParentHBSD(1:datasetSize*0.8,:);...
    rsrpMatParentHBSE(1:datasetSize*0.2,:);];
  optLabels     = [txBeamMatEBSD(1:datasetSize*0.8,:);...
    txBeamMatEBSE(1:datasetSize*0.2,:)];
  optLabelsIdx  = [txBeamIdEBSD(1:datasetSize*0.8);...
    txBeamIdEBSE(1:datasetSize*0.2)];
  rsrpMatHBS  = [rsrpMatChildHBSD(1:datasetSize*0.8,:);...
    rsrpMatChildHBSE(1:datasetSize*0.2,:)];
  txBeamMatChildHBS = [txBeamMatChildHBSD(1:datasetSize*0.8,:);...
    txBeamMatChildHBSE(1:datasetSize*0.2,:)];
  rsrpMatEBS = [rsrpMatEBSD(1:datasetSize*0.8,:); ...
    rsrpMatEBSE(1:datasetSize*0.2,:)];
end



%% Proposed Model
NNType = 'SLNN.py';

torch         = py.importlib.import_module("torch");
inputData     = py.torch.from_numpy(py.numpy.array(inputFeat)).float;
inputLabels   = py.torch.from_numpy(py.numpy.array(optLabels)).float;


[nnPredictions, ~, ~, ~, ~, ~, ~,...
  ~, ~, ~, ]  = pyrunfile(NNType, ["predictionsNp", ...
  "trainIdxNp", "validIdxNp", "testIdxNp", "trainLossHistoryNp", "valLossHistoryNp", "testLossHistoryNp",...
  "trainAccHistoryNp", "valAccHistoryNp", "testAccHistoryNp"], ...
  inputData=inputData, inputLabels=inputLabels);

slNNPredictions = double(nnPredictions);

%% FCNN from [5]
NNType = 'FCNN.py';

[nnPredictions, ~, ~, ~, ~, ~, ~,...
  ~, ~, ~, ]  = pyrunfile(NNType, ["predictionsNp", ...
  "trainIdxNp", "validIdxNp", "testIdxNp", "trainLossHistoryNp", "valLossHistoryNp", "testLossHistoryNp",...
  "trainAccHistoryNp", "valAccHistoryNp", "testAccHistoryNp"], ...
  inputData=inputData, inputLabels=inputLabels);

fcNNPredictions = double(nnPredictions);

%% CNN from [6]
NNType = 'SupResCNN.py';


[nnPredictions, trainIdx, validIdx, testIdx, ~, ~, ~,...
  ~, ~, ~, ]  = pyrunfile(NNType, ["predictionsNp", ...
  "trainIdxNp", "validIdxNp", "testIdxNp", "trainLossHistoryNp", "valLossHistoryNp", "testLossHistoryNp",...
  "trainAccHistoryNp", "valAccHistoryNp", "testAccHistoryNp"], ...
  inputData=inputData, inputLabels=inputLabels);

trainInputIdxs  = double(trainIdx);
validInputIdxs  = double(validIdx);
testInputIdxs   = double(testIdx);

trainInput  = inputFeat(trainInputIdxs+1,:);
trainlabelsIdx = optLabelsIdx(trainInputIdxs+1,:);
validInput  = inputFeat(validInputIdxs+1,:);
validLabelsIdx = optLabelsIdx(validInputIdxs+1,:);
testInput   = inputFeat(testInputIdxs+1,:);
testLabelsIdx  = optLabelsIdx(testInputIdxs+1,:);

srCNNPredictions = double(nnPredictions);
testDataLength = size(testLabelsIdx, 1);

%%
testLabelsMatChildHBS = txBeamMatChildHBS(testInputIdxs+1,:);
[testLabelsIdxChildHBS,~] = find(testLabelsMatChildHBS.');

K = 4;
accHBS    = zeros(1,K);
accSLNN   = zeros(1,K);
accFCNN   = zeros(1,K);
accSRCNN  = zeros(1,K);

for k = 1:K
  predCorrectHBS    = zeros(testDataLength,1);
  predCorrectSLNN = zeros(testDataLength,1);
  predCorrectFCNN = zeros(testDataLength,1);
  predCorrectSRCNN = zeros(testDataLength,1);


  for n = 1:testDataLength
    trueOptBeamIdx = testLabelsIdx(n);

    % HBS
    [~, topKIdxHBS] = maxk(testLabelsMatChildHBS(n, :),1);
    if sum(topKIdxHBS == trueOptBeamIdx) > 0
      % if true, then the true correct index belongs to one of the K predicted indices
      predCorrectHBS(n,1) = 1;
    end

    % Neural Network
    [~, topKIdxSLNN] = maxk(slNNPredictions(n, :),k);
    if sum(topKIdxSLNN == trueOptBeamIdx) > 0
      % if true, then the true correct index belongs to one of the K predicted indices
      predCorrectSLNN(n,1) = 1;
    end

    [~, topKIdxFCNN] = maxk(fcNNPredictions(n, :),k);
    if sum(topKIdxFCNN == trueOptBeamIdx) > 0
      % if true, then the true correct index belongs to one of the K predicted indices
      predCorrectFCNN(n,1) = 1;
    end


    [~, topKIdxSRCNN] = maxk(srCNNPredictions(n, :),k);
    if sum(topKIdxSRCNN == trueOptBeamIdx) > 0
      % if true, then the true correct index belongs to one of the K predicted indices
      predCorrectSRCNN(n,1) = 1;
    end

  end

  accHBS(k)   = sum(predCorrectHBS)/testDataLength*100;
  accSLNN(k)        = sum(predCorrectSLNN)/testDataLength*100;
  accFCNN(k)        = sum(predCorrectFCNN)/testDataLength*100;
  accSRCNN(k)        = sum(predCorrectSRCNN)/testDataLength*100;
end

% save('accSRCNN.mat', 'accKNN', "accNN", "accHBS")
%% RSRP

rsrpMatEbsTest = rsrpMatEBS(testInputIdxs+1,:);
rsrpMatHbsTest = rsrpMatHBS(testInputIdxs+1,:);
testLabelsMatChildHBS = txBeamMatChildHBS(testInputIdxs+1,:);
[testLabelsIdxChildHBS,~] = find(testLabelsMatChildHBS.');

rsrpEBS = zeros(1,K);
rsrpHBS = zeros(1,K);
rsrpSLNN = zeros(1,K);
rsrpFCNN = zeros(1,K);
rsrpSRCNN = zeros(1,K);

for k = 1:K
  rsrpSumEBS  = 0;
  rsrpSumHBS  = 0;
  rsrpSumSLNN   = 0;
  rsrpSumFCNN   = 0;
  rsrpSumSRCNN   = 0;


  for n = 1:testDataLength

    % Exhaustive Search
    trueOptBeamIdx = testLabelsIdx(n);
    rsrpEbs = rsrpMatEbsTest(n,:);
    rsrpSumEBS = rsrpSumEBS + rsrpEbs(trueOptBeamIdx);

    % Hierarchical Search
    trueOptBeamIdxHbs = testLabelsIdxChildHBS(n);
    rsrpHbs = rsrpMatHbsTest(n,:);
    rsrpSumHBS = rsrpSumHBS + rsrpHbs(trueOptBeamIdxHbs);

    % Neural Network
    [~, topKIdxSLNN] = maxk(slNNPredictions(n, :),k);
    rsrpSumSLNN = rsrpSumSLNN + max(rsrpEbs(topKIdxSLNN));

    [~, topKIdxFCNN] = maxk(fcNNPredictions(n, :),k);
    rsrpSumFCNN = rsrpSumFCNN + max(rsrpEbs(topKIdxFCNN));

    [~, topKIdxSRCNN] = maxk(srCNNPredictions(n, :),k);
    rsrpSumSRCNN = rsrpSumSRCNN + max(rsrpEbs(topKIdxSRCNN));


  end
  rsrpEBS(k)  = rsrpSumEBS/testDataLength;
  rsrpHBS(k)  = rsrpSumHBS/ testDataLength;
  rsrpSLNN(k)   = rsrpSumSLNN/testDataLength;
  rsrpFCNN(k)   = rsrpSumFCNN/testDataLength;
  rsrpSRCNN(k)   = rsrpSumSRCNN/testDataLength;
end



%% Accuracy vs Overhead
figure(3)
accEBS = 100;
set(0,'DefaultAxesFontSize',8); %Eight point Times is suitable typeface for an IEEE paper. Same as figure caption size
set(0,'DefaultFigureColor','w')
set(0,'defaulttextinterpreter','tex') %Allows us to use LaTeX maths notation
set(0, 'DefaultAxesFontName', 'times');
set(gcf, 'Units','centimeters')
set(gcf, 'Position',[0 0 8.89 8]) %Absolute print dimensions of figure. 8.89cm is essential here as it is the linewidth of a column in IEEE format

figure(3);b = bar([ accEBS-mean(accHBS)], 'FaceColor',[0.58 0.39 0.39]);hold on;
b.BarWidth = 0.2;
grid on

x = [ 0  0 0 ; accEBS-accSRCNN(4) accEBS-accFCNN(4) accEBS-accSLNN(4) ; accEBS-accSRCNN(2) accEBS-accFCNN(2) accEBS-accSLNN(2)   ; accEBS-accSRCNN(1) accEBS-accFCNN(1) accEBS-accSLNN(1)];
figure (3); y = bar(x); hold on;
y(1).FaceColor = [0 0.45 0.74];
y(2).FaceColor = [0.85 0.33 0.1];
y(3).FaceColor = [0.93 0.69 0.13];

xticks([1 2 3 4 5])
xticklabels({'HBS-20','ML Top-4','ML Top-2','ML Top-1'})

yyaxis left
ylabel('Beam Prediction Error [%]')
yyaxis right
ylabel('Beam Measurement Overhead [%]')
ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = '[0 0.498039215803146 0]';

Y=[20/64, 20/64, 18/64, 16/64].*100;
figure(3); plot(Y, 'Color',[0 0.498039215803146 0], 'Marker','o', 'MarkerFaceColor', 'g', LineWidth=1);hold on
legend('HBS','CNN [6]','FC-NN [5]', 'Proposed model', 'Measurement Overhead')
ylim([0 36])

%% RSRP Gain
figure(4)
set(0,'DefaultAxesFontSize',8); %Eight point Times is suitable typeface for an IEEE paper. Same as figure caption size
set(0,'DefaultFigureColor','w')
set(0,'defaulttextinterpreter','tex') %Allows us to use LaTeX maths notation
set(0, 'DefaultAxesFontName', 'times');

% figure  %Let's make a simple time series plot of notional data
set(gcf, 'Units','centimeters')

%Set figure total dimension
set(gcf, 'Position',[0 0 8.89 6]) %Absolute print dimensions of figure. 8.89cm is essential here as it is the linewidth of a column in IEEE format
%Height can be adusted as suits, but try and be consistent amongst figures for neatness
%[pos_from_left, pos_from_bottom, fig_width, fig_height]

lineWidth = 1;
plot(1:K,rsrpEBS(end,:),'LineWidth',lineWidth, 'Color',[0.45 0.26 0.26], 'Marker','o', 'MarkerFaceColor',[0.45 0.26 0.26]);
hold on
plot(1:K,rsrpHBS(end,:),'LineWidth',lineWidth, 'Color',[0.58 0.39 0.39], 'Marker','*', 'MarkerFaceColor',[0.58 0.39 0.39]);
plot(1:K,rsrpSRCNN(end,:),'LineWidth',lineWidth, 'Color', [0 0.45 0.74], 'Marker' , 'square'  , 'MarkerFaceColor',[0 0.45 0.74])
plot(1:K,rsrpFCNN(end,:),'LineWidth',lineWidth, 'Color', [0.85 0.33 0.1], 'Marker' , 'diamond', 'MarkerFaceColor',[0.85 0.33 0.1])
plot(1:K,rsrpSLNN(end,:),'LineWidth',lineWidth, 'Color', [0.93 0.69 0.13], 'Marker' , 'pentagram', 'MarkerFaceColor',[0.93 0.69 0.13])

grid on
xticks([1 2 3 4])
xticklabels({'ML Top-1','ML Top-2','ML Top-3','ML Top-4'})
ylabel('Average RSRP [dBm]')
legend('EBS','HBS','CNN [6]','FC-NN [5]', 'Proposed model')
k = 1:4;


