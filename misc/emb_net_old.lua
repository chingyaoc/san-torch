require 'nn'
local utils = require 'misc.utils'
local LanguageEmbedding = require 'misc.LanguageEmbedding'

local layer, parent = torch.class('nn.emb_net', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.emb_size = utils.getopt(opt, 'emb_size')
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    local dropout = utils.getopt(opt, 'dropout', 0)
    self.batch_size = utils.getopt(opt, 'batch_size')
    self.seq_length = utils.getopt(opt, 'seq_length')

    -- question embedding
    self.LE = LanguageEmbedding.LE(self.vocab_size, self.hidden_size, self.hidden_size, self.seq_length)  

    self.core = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout)

    -- image feature embedding
    self.cnn = nn.Sequential()
                      :add(nn.View(512):setNumInputDims(2))
                      :add(nn.Linear(512, self.hidden_size))
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
                      :add(nn.Dropout(0.5))

    self:_createInitState(self.batch_size)
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
    print('constructing clones inside the ques_level')
    self.cores = {self.core}
    for t=1,self.seq_length do
        self.cores[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.cores == nil then self:createClones(); return; end
    print('resharing clones inside the ques_level')
    self.cores[1] = self.core
    for t=1,self.seq_length do
        self.cores[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:getModulesList()
    return {self.LE, self.core, self.cnn}
end

function layer:parameters()
    local p1,g1 = self.LE:parameters()
    local p2,g2 = self.core:parameters()
    local p3,g3 = self.cnn:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p2) do table.insert(params, v) end
    for k,v in pairs(p3) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end
    for k,v in pairs(g3) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.LE:training()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:training() end
    self.cnn:training()
end

function layer:evaluate()
    self.LE:evaluate()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:evaluate() end
    self.cnn:evaluate()
end

--[[
function layer:updateOutput(input)
  local ques = input[1]
  local img = input[2]
  local seq_len = input[3]

  if self.cores == nil then self:createClones() end -- lazily create clones on first forward pass
  local batch_size = ques:size(1)
  self.tmax = torch.max(seq_len)
  self.tmin = torch.min(seq_len)

  self:_createInitState(batch_size)
  self.fore_state = {[0] = self.init_state}
  self.fore_inputs = {}
  self.core_output:resize(batch_size, self.seq_length, self.rnn_size):zero()

  -- quesiton: one-hot -> embedding dim
  self.embed_output = self.LE:forward(ques)

  -- quesiton: LSTM encoding
  for t=1,self.tmax do
      self.fore_inputs[t] = {self.embed_output:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(self.fore_state[t-1])}
      local out = self.cores[t]:forward(self.fore_inputs[t])
      if t > self.tmin then
        for i=1,self.num_state+1 do
          out[i]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(out[i]), 0)
        end
      end
      self.fore_state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.fore_state[t], out[i]) end
      
      self.core_output:narrow(2,t,1):copy(out[self.num_state+1])
  end  

  -- final hidden state
  self.lstm_output:resize(batch_size, 1, self.rnn_size):zero()
  for i=1,self.batch_size do
      self.lstm_output:narrow(1,i,1):copy((self.core_output:narrow(1,i,1)):narrow(2,seq_len[i],1))
  end
  self.lstm_output:resize(batch_size, self.rnn_size)

  -- image: feature embedding
  self.img_feat = self.cnn:forward(img)

  return {self.lstm_output, self.img_feat}
end


function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local img = input[2]
  local seq_len = input[3]  

  local d_ques_feat = gradOutput[1]
  local d_img_feat = gradOutput[2]

  local batch_size = ques:size(1)

  local d_core_state = {[self.tmax] = self.init_state} -- initial dstates
  local d_embed_core = d_embed_core or self.core_output:new()
  d_embed_core:resize(batch_size, self.seq_length, self.rnn_size):zero()

  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#d_core_state[t] do table.insert(dout, d_core_state[t][k]) end
    table.insert(dout, d_core_output:narrow(2,t,1):contiguous():view(-1, self.hidden_size))
    local dinputs = self.cores[t]:backward(self.fore_inputs[t], dout)

    if t > self.tmin then
      for k=1,self.num_state+1 do
        dinputs[k]:maskedFill(self.mask:narrow(2,t,1):contiguous():view(batch_size,1):expandAs(dinputs[k]), 0)
      end
    end
    d_core_state[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
    d_embed_core:narrow(2,t,1):copy(dinputs[1])
  end


  local dummy = self.LE:backward(seq, d_embed_core)

  local d_imgfeat = self.cnn:backward(img, d_img_feat)
  self.gradInput = d_imgfeat
  
  return self.gradInput
end
--]]
