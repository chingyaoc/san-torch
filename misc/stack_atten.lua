require 'nn'
local utils = require 'misc.utils'
local attention = require 'misc.attention'

local layer, parent = torch.class('nn.stack_atten', 'nn.Module')
function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.emb_size = utils.getopt(opt, 'emb_size')
    self.hidden_size = utils.getopt(opt, 'hidden_size')
    self.att_size = utils.getopt(opt, 'att_size')
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    self.dropout = utils.getopt(opt, 'dropout', 0.5)
    self.batch_size = utils.getopt(opt, 'batch_size')
    self.output_size = utils.getopt(opt, 'output_size')
    self.seq_length = utils.getopt(opt, 'seq_length')


    self.atten_encode = attention.stack_atten(self.hidden_size, self.att_size,196,self.output_size,0.5)
end

function layer:getModulesList()
    return {self.atten_encode}
end

function layer:parameters()
    local p1,g1 = self.atten_encode:parameters()
    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    self.atten_encode:training()
end

function layer:evaluate()
    self.atten_encode:evaluate()
end

function layer:updateOutput(input)
  local ques_feat = input[1]
  local img_feat = input[2]

  local out_feat = self.atten_encode:forward({ques_feat, img_feat})

  return out_feat
end

function layer:updateGradInput(input, gradOutput)
  local ques_feat = input[1]
  local img_feat = input[2]

  local d_ques_feat, d_img_feat = unpack(self.atten_encode:backward({ques_feat, img_feat}, gradOutput))

  self.gradInput = {d_ques_feat, d_img_feat}

  return self.gradInput
end
