------------------------------------------------------------------------------------
--  Torch Implementation of Stacked Attention Networks for Image Question Answering
--  Ching-Yao Chuang
--  National Tsing Hua University
--  if you have any question about the code, please contact james847286@gmail.com
------------------------------------------------------------------------------------

require 'nn'
require 'torch'
--require 'optim'
require 'misc.DataLoaderDisk'
require 'misc.emb_net'
require 'misc.attention'
require 'misc.stack_atten'
require 'misc.optim_updates'
local utils = require 'misc.utils'
require 'xlua'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Eval a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_train_h5','data/vqa_data_img_vgg_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1234, 'random number generator seed to use')

cmd:text()

local batch_size = 256

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then
  require 'cudnn'
  end
  --cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if string.len(opt.start_from) > 0 then
  loaded_checkpoint = torch.load(opt.start_from)
  lmOpt = loaded_checkpoint.lmOpt
else
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.emb_size = opt.emb_size
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.att_size = opt.att_size
  lmOpt.num_layers = opt.rnn_layers
  lmOpt.dropout = 0.5
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.output_size
  lmOpt.feature_type = opt.feature_type
end

protos.emb = nn.emb_net(lmOpt)
protos.atten = nn.stack_atten(lmOpt)
protos.crit = nn.CrossEntropyCriterion()

print('ship everything to GPU...')
-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local eparams, grad_eparams = protos.emb:getParameters()
local aparams, grad_aparams = protos.atten:getParameters()

eparams:uniform(-0.01, 0.01)
aparams:uniform(-0.01, 0.01)

if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  eparams:copy(loaded_checkpoint.eparams)
  aparams:copy(loaded_checkpoint.aparams)
end

print('total number of parameters in embedding net: ', eparams:nElement())
assert(eparams:nElement() == grad_eparams:nElement())

print('total number of parameters in stacked attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

collectgarbage()

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------

local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type}

collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.emb:evaluate()
  protos.atten:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_evals = 0  
  local predictions = {}
  local total_num = loader:getDataNum(2)
  print(total_num)
  local logprob_all = torch.Tensor(total_num, 1000)
  local ques_id = torch.Tensor(total_num)

  for i = 1, total_num, batch_size do
    xlua.progress(i, total_num)
    local r = math.min(i+batch_size-1, total_num) 

    local data = loader:getBatch{batch_size = r-i+1, split = split}
    -- ship the data to cuda
    if opt.gpuid >= 0 then
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
    end

    if r == total_num then
      image_pad = torch.CudaTensor(i+256-1-r, data.images:size(2), data.images:size(3)):fill(1)
      questions_pad = torch.CudaTensor(i+256-1-r, data.questions:size(2)):fill(1)
      ques_len_pad = torch.CudaTensor(i+256-1-r):fill(1)

      data.images = torch.cat(data.images, image_pad, 1)
      data.questions = torch.cat(data.questions, questions_pad, 1)
      data.ques_len = torch.cat(data.ques_len, ques_len_pad, 1)
    end

    local ques_feat, img_feat = unpack(protos.emb:forward({data.questions, data.images, data.ques_len}))

    local pred_answer = protos.atten:forward({ques_feat, img_feat})
    
    if r == total_num then
      pred_answer = pred_answer:sub(1,r-i+1)
    end

   
    logprob_all:sub(i, r):copy(pred_answer:float())
    ques_id:sub(i, r):copy(data.ques_id)

    end


    tmp,pred=torch.max(logprob_all,2);

    for i=1,total_num do
        local ans = loader.ix_to_ans[tostring(pred[{i,1}])]
        table.insert(predictions,{question_id=ques_id[i],answer=ans})
    end

  return {predictions}
end

predictions = eval_split(2)

utils.write_json('OpenEnded_mscoco_lstm_results.json', predictions[1])





