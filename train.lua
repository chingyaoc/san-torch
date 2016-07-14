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
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_train_h5','data/vqa_data_img_vgg_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

cmd:option('-emb_size',500,'the size after embeeding from onehot')
cmd:option('-hidden_size',1024,'the hidden layer size of the model.')
cmd:option('-rnn_size',1024,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-att_size',512,'size of sttention vector which refer to k in paper')
cmd:option('-batch_size',200,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-output_size', 1000, 'number of output answers')
cmd:option('-rnn_layers',1,'number of the rnn layer')


-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-momentum',0.9,'momentum')
cmd:option('-learning_rate_decay_start', 100, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 1500, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1250)

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 6000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'save/train_vgg', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 600, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 2, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1234, 'random number generator seed to use')

cmd:text()

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
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type}

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if string.len(opt.start_from) > 0 then
  local start_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type ,  opt.start_from)
  loaded_checkpoint = torch.load(start_path)
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
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.emb:evaluate()
  protos.atten:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local predictions = {}
  local total_num = loader:getDataNum(split)
  while true do
    local data = loader:getBatch{batch_size = opt.batch_size, split = split}
    -- ship the data to cuda
    if opt.gpuid >= 0 then
      data.answer = data.answer:cuda()
      data.images = data.images:cuda()
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
    end
  n = n + data.images:size(1)
  xlua.progress(n, total_num)
 
  local ques_feat, img_feat = unpack(protos.emb:forward({data.questions, data.images, data.ques_len}))

  local pred_answer = protos.atten:forward({ques_feat, img_feat})
  -- forward the language model criterion
  local loss = protos.crit:forward(pred_answer, data.answer)
 
    local tmp,pred=torch.max(pred_answer,2)

    for i = 1, pred:size()[1] do

      if pred[i][1] == data.answer[i] then
        right_sum = right_sum + 1
      end
    end

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    if n >= total_num then break end
  end

  return loss_sum/loss_evals, right_sum / total_num
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.emb:training()
  grad_eparams:zero()  

  protos.atten:training()
  grad_aparams:zero()

  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
  if opt.gpuid >= 0 then
    data.answer = data.answer:cuda()
    data.questions = data.questions:cuda()
    data.ques_len = data.ques_len:cuda()
    data.images = data.images:cuda()
  end

  local ques_feat, img_feat = unpack(protos.emb:forward({data.questions, data.images, data.ques_len}))

  local pred_answer = protos.atten:forward({ques_feat, img_feat})

  -- forward the language model criterion
  local loss = protos.crit:forward(pred_answer, data.answer)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(pred_answer, data.answer)

  local d_ques_feat, d_img_feat = unpack(protos.atten:backward({ques_feat, img_feat}, dlogprobs))

  local dummy = protos.emb:backward({data.questions, data.images, data.ques_len}, {d_ques_feat, d_img_feat})
  -----------------------------------------------------------------------------
  -- and lets get out!

  running_loss = loss

  local stats = {}
  stats.dt = dt
  local losses = {}
  losses.total_loss = loss
  return losses, stats
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------

local loss0
local e_optim_state = {}
local a_optim_state = {}
local loss_history = {}
local accuracy_history = {}
local learning_rate_history = {}
local best_val_loss = 10000
local grad_clip = 0.1
local ave_loss = 0
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
--local decay_factor = 0.999
local learning_rate = opt.learning_rate
-- create the path to save the model.
paths.mkdir(opt.checkpoint_path .. '_' .. 'MODEL')

while true do
  -- eval loss/gradient
  local losses, stats = lossFun()

  ave_loss = ave_loss + losses.total_loss
  -- decay the learning rate
  
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end
  
  --print(string.format('iter %d: current loss: %f, lr: %f', iter, running_loss, learning_rate))

  if iter % opt.losses_log_every == 0 then
    ave_loss = ave_loss / opt.losses_log_every
    loss_history[iter] = losses.total_loss 
    accuracy_history[iter] = ave_loss
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, losses.total_loss, ave_loss, learning_rate, timer:time().real))

    ave_loss = 0
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
      local val_loss, val_accu = eval_split(2)
      print('validation loss: ', val_loss, 'accuracy ', val_accu)

      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. 'MODEL', 'model_id' .. opt.id .. '_iter'.. iter)
      torch.save(checkpoint_path .. '.t7', {eparams=eparams, aparams=aparams, lmOpt=lmOpt}) 

      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.accuracy_history = accuracy_history
      checkpoint.learning_rate_history = learning_rate_history

      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. 'MODEL', 'checkpoint' .. '.json')

      utils.write_json(checkpoint_path, checkpoint)
      print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

  end

  -- gradient clipping
  --grad_eparams:clamp(-10,10)
  --grad_aparams:clamp(-10,10)

  -- perform a parameter update
  if opt.optim == 'sgd' then
    sgdm(eparams, grad_eparams, learning_rate, opt.momentum, e_optim_state)
    sgdm(aparams, grad_aparams, learning_rate, opt.momentum, a_optim_state)
  elseif opt.optim == 'rmsprop' then
    rmsprop(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
    rmsprop(aparams, grad_aparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, a_optim_state)
  else
    error('bad option opt.optim')
  end

  iter = iter + 1
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end
