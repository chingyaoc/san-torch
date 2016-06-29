require 'nngraph'
require 'nn'
local mask = {}

function mask.masking(batch_size, hidden_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local lstm_feat = inputs[1]                                 -- [batch_size, max_len, dim]
    local mask = inputs[2]                                      -- [batch_size, max_len]

    local mask_expand = nn.View(1,-1):setNumInputDims(1)(mask)  -- [batch_size, 1, max_len]
    local final_h_expand = nn.MM(false, false)({mask_expand, lstm_feat})       -- [batch_size, 1, dim]
    local final_h= nn.View(-1, hidden_size)(final_h_expand)              -- [batch_size, dim]

    table.insert(outputs, final_h)

    return nn.gModule(inputs, outputs)
end

return mask
