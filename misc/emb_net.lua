require 'nn'
local utils = require 'misc.utils'
require 'misc.LSTM'

local layer, parent = torch.class('nn.emb_net', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.emb_size = utils.getopt(opt, 'emb_size')
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    self.dropout = utils.getopt(opt, 'dropout', 0)
    self.seq_length = utils.getopt(opt, 'seq_length')

    -- LSTM
    local V, D, H = self.vocab_size, self.emb_size, self.hidden_size

    self.net = nn.Sequential()
    self.rnns = {}
    self.bn_view_in = {}
    self.bn_view_out = {}
    -- Word Embedding
    self.net:add(nn.LookupTable(V, D))
    self.net:add(nn.Tanh())
    self.net:add(nn.Dropout(self.dropout))
    -- LSTM Encoding
    for i = 1, self.num_layers do
      local prev_dim = H
      if i == 1 then prev_dim = D end
      local rnn
      rnn = nn.LSTM(prev_dim, H)
      rnn.remember_states = true
      table.insert(self.rnns, rnn)
      self.net:add(rnn)
      if self.dropout > 0 then
        self.net:add(nn.Dropout(self.dropout))
      end
    end

    self.masking = masking(self.hidden_size)
 
    -- image feature embedding
    self.cnn = nn.Sequential()
                      :add(nn.View(512):setNumInputDims(2))
                      :add(nn.Linear(512, self.hidden_size))
                      :add(nn.View(-1, 196, self.hidden_size))
                      :add(nn.Tanh())
                      :add(nn.Dropout(0.5))
end

function layer:getModulesList()
    return {self.net, self.masking, self.cnn}
end

function layer:parameters()
    local p1,g1 = self.net:parameters()
    local p2,g2 = self.masking:parameters()
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
    self.net:training()
    self.masking:training()
    self.cnn:training()
end

function layer:evaluate()
    self.net:evaluate()
    self.masking:evaluate()
    self.cnn:evaluate()
end

function masking(hidden_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local lstm_feat = inputs[1] 				-- [batch_size, max_len, dim]
    local mask = inputs[2]  					-- [batch_size, max_len]

    local mask_expand = nn.View(1,-1):setNumInputDims(1)(mask)  -- [batch_size, 1, max_len] 
    local final_h_expand = nn.MM(false, false)({mask_expand, lstm_feat})       -- [batch_size, 1, dim]
    local final_h= nn.View(-1, hidden_size)(final_h_expand)              -- [batch_size, dim]

    table.insert(outputs, final_h)

    return nn.gModule(inputs, outputs)
end


function layer:updateOutput(input)
  local ques = input[1]
  local img = input[2]
  local seq_len = input[3]

  local batch_size = ques:size(1)

  self.lstm_output = self.net:forward(ques)
   
  self.max_len = ques:size(2)
  local len = torch.LongTensor():resize(seq_len:size()):copy(seq_len)
  local mask_zero = torch.Tensor(batch_size,self.max_len):fill(0)
  self.mask = mask_zero:scatter(2,len:view(-1,1),1)		-- [batch_size, max_len]
  -- Make Sure that every tensor have the same type
  self.mask = self.mask:cuda()

  local final_h = self.masking:forward({self.lstm_output, self.mask})

  self.img_feat = self.cnn:forward(img)

  return {final_h, self.img_feat}
end


function layer:updateGradInput(input, gradOutput)
  local ques = input[1]
  local img = input[2]
  local seq_len = input[3]  

  local batch_size = ques:size(1)

  local d_ques_feat = gradOutput[1]
  local d_img_feat = gradOutput[2]
  local batch_size = ques:size(1)
  
  local d_lstm_output, d_mask = unpack(self.masking:backward({self.lstm_output, self.mask}, d_ques_feat))

  local d_ques = self.net:backward(ques, d_lstm_output)

  local d_img = self.cnn:backward(img, d_img_feat)
  self.gradInput = d_img

  return self.gradInput
end
