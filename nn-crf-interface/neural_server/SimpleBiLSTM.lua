local SimpleBiLSTM, parent = torch.class('SimpleBiLSTM', 'AbstractNeuralNetwork')

function SimpleBiLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function SimpleBiLSTM:defineGlobalString()
    self.padToken = "<PAD>"
    self.unkToken = "<UNK>"
    self.startToken = "<START>"
    self.endToken = "<END>"
    self.e1Start = "<e1>"
    self.e1End = "</e1>"
    self.e2Start = "<e2>"
    self.e2End = "</e2>"
end

function SimpleBiLSTM:loadEmbObj()
    local data = self.data
    self.embeddingSize = data.embeddingSize
    if data.embedding == 'google' then
        self.embeddingObject = loadGoogleEmbObj()
        self.embeddingSize = 300
    elseif data.embedding == 'turian' then
        self.embeddingObject = loadTurianEmbObj()
        self.embeddingSize = 50
    elseif data.embedding == 'glove' then
        self.embeddingObject = loadGloveEmbObj()
        self.embeddingSize = 100
    elseif data.embedding == 'random' then 
        print("using random embedding")
    else
        error('unknown embedding type: '.. data.embedding)
    end
end

function SimpleBiLSTM:initialize(javadata, ...)
    self.data = {}
    self:defineGlobalString()
    local data = self.data
    data.sentences = listToTable(javadata:get("nnInputs"))
    data.embeddingSize = javadata:get("embeddingSize")
    self.numLabels = javadata:get("numLabels")
    data.embedding = javadata:get("embedding")
    self.dropout = javadata:get("dropout")
    self.fixEmbedding = javadata:get("fixEmbedding")
    local modelPath = javadata:get("nnModelFile")
    local isTraining = javadata:get("isTraining")
    data.isTraining = isTraining

    if isTraining then
        self:loadEmbObj()
        self.x = self:prepare_input(isTraining)
        self.numSent = #data.sentences
    end

    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        print(self.net)
    end

    if self.net == nil then 
        self:load_model(modelPath)
    end

    if not isTraining then 
        self.testInput = self:prepare_input(isTraining)
    end
    self.output = torch.Tensor()
    self.gradOutput = torch.Tensor()
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

function SimpleBiLSTM:buildBiGRU(inputSize, outputSize, dropout, tanhGRU)
    local bigru = nn.Sequential():add(nn.Transpose({1,2})):add(nn.SplitTable(1))
    -- local fwdSeq = nn.Sequencer(nn.GRU(inputSize, outputSize, 9999, dropout):maskZero(1))
    local fwdSeq = nn.Sequencer(nn.FastLSTM(inputSize, outputSize):maskZero(1))
    local bwdSeq = nn.Sequential():add(nn.ReverseTable())
    -- bwdSeq:add(nn.Sequencer(nn.GRU(inputSize, outputSize, 9999, dropout):maskZero(1)))
    bwdSeq:add(nn.Sequencer(nn.FastLSTM(inputSize, outputSize):maskZero(1)))
    bwdSeq:add(nn.ReverseTable())
    local biconcat = nn.ConcatTable():add(fwdSeq):add(bwdSeq)
    bigru:add(biconcat):add(nn.ZipTable()):add(nn.Sequencer(nn.JoinTable(1,1)))
    local mapTable = nn.MapTable()
    local combineSize
    if tanhGRU then
        local mapOp = nn.Sequential()
        mapOp:add(nn.Linear(2 * outputSize, outputSize)):add(nn.Tanh())
        mapOp:add(nn.Unsqueeze(1))
        mapTable:add(mapOp)
        combineSize = outputSize
    else
        mapTable:add(nn.Unsqueeze(1))
        combineSize = 2 * outputSize
    end
    bigru:add(mapTable):add(nn.JoinTable(1)):add(nn.Transpose({1,2}))
    return bigru
end

--The network is only created once is used.
function SimpleBiLSTM:createNetwork()
    local data = self.data
    local hiddenSize = data.hiddenSize
    local embeddingSize = self.embeddingSize
    local layer2hiddenSize = data.layer2hiddenSize
    local gruHiddenSize = self.gruHiddenSize
    local sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, embeddingSize)
    if not data.embedding == 'random' then
        for i =1, self.vocabSize do
            sharedLookupTable.weight[i+1]:copy(self.embeddingObject:word2vec(self.idx2word[i]))
        end
    end
    self.lt = sharedLookupTable
    print("Word Embedding layer: " .. self.lt.weight:size(1) .. " x " .. self.lt.weight:size(2))
    if self.fixEmbedding then
        self.lt.accGradParameters = function() end
        self.lt.parameters = function() end
    end

    local brnn = self:buildBiGRU(embeddingSize, embeddingSize, self.dropout, true)
    
    local net = nn.Sequential()
    net:add(self.lt)
    -- if self.embDropout > 0 then
    --     net:add(nn.Dropout(self.embDropout)) --adding lookup table
    -- end
    net:add(brnn) -- go into brnn
    self.net = net
    if self.gpuid >= 0 then
        self.net:cuda() 
    end
end


function SimpleBiLSTM:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    if self.doOptimization then
        self:createOptimizer()
        -- no return array if optim is done here
    else
        if self.gpuid >= 0 then
            -- since the the network is gpu network.
            self.paramsDouble = self.params:double()
            self.paramsDouble:retain()
            self.params:retain()
            self.paramsPtr = torch.pointer(self.paramsDouble)
            self.gradParamsDouble = self.gradParams:double()
            self.gradParamsDouble:retain()
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParamsDouble)
            return self.paramsPtr, self.gradParamsPtr
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    end
end

function SimpleBiLSTM:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization then
        self.params:copy(self.paramsDouble:cuda())
    end
    if isTraining then
        self.net:training()
    else
        self.net:evaluate()
    end
    local nnInput = self:getForwardInput(isTraining, batchInputIds)
    local lstmOutput
    if isTraining then
        lstmOutput = self.net:forward(nnInput)
    else
        lstmOutput = torch.Tensor()
        if self.gpuid >=0 then lstmOutput = lstmOutput:cuda() end
        local instSize = nnInput:size(1) --number of sentences 
        local testBatchSize = 10   ---test batch size = 10
        for i = 1, instSize, testBatchSize do
            if i + testBatchSize - 1 > instSize then testBatchSize =  instSize - i + 1 end
            local tmpOut = self.net:forward(nnInput:narrow(1, i, testBatchSize))
            lstmOutput = torch.cat(lstmOutput, tmpOut, 1)
        end
    end
    if self.gpuid >= 0 then
        lstmOutput = lstmOutput:double()
    end 
    self.output = lstmOutput
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
end

function SimpleBiLSTM:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            batchInputIds:add(1) -- because the sentence is 0 indexed.
            self.batchInputIds = batchInputIds
            self.batchInput = self.x:index(1, batchInputIds)
            return self.batchInput
        else
            return self.x
        end
    else
        return self.testInput
    end
end

function SimpleBiLSTM:getBackwardInput()
    if self.batchInputIds ~= nil then
        return self.batchInput
    else
        return self.x
    end
end

function SimpleBiLSTM:getBackwardSentNum()
    if self.batchInputIds ~= nil then
        return self.batchInputIds:size(1)
    else
        return self.numSent
    end
end

function SimpleBiLSTM:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    local backwardInput = self:getBackwardInput()  --since backward only happen in training
    self.gradOutput = gradOutputTensor
    if self.gpuid >= 0 then
        self.gradOutput = self.gradOutput:cuda()
    end
    self.net:training()
    self.net:backward(backwardInput, self.gradOutput)

    if self.gpuid >= 0 then
        self.gradParamsDouble:copy(self.gradParams:double())
    end
end

function SimpleBiLSTM:prepare_input()
    local data = self.data
    local sentences = data.sentences
    local sentence_toks = {}
    local maxLen = 0
    for i=1,#sentences do
        local sentence = sentences[i]
        local tokens = stringx.split(sentence," ")
        table.insert(sentence_toks, tokens)
        if #tokens > maxLen then
            maxLen = #tokens
        end
    end

    --note that inside if the vocab is already created
    --just directly return
    self:buildVocab(sentences, sentence_toks)    
    ---build tensor input
    local inputs = torch.IntTensor(#sentences, maxLen)
    self:fillInputs(#sentences, inputs, maxLen, sentence_toks)
    if self.gpuid >= 0 then 
        inputs = inputs:cuda()
    end
    print("number of sentences: "..#sentences)
    print("max sentence length: "..maxLen)
    return inputs
end

function SimpleBiLSTM:fillInputs(numSents, inputTensor, maxLen, toks)
    for sId=1,numSents do
        local tokens = toks[sId]
        for step=1,maxLen do
            if step > #tokens then
                inputTensor[sId][step] = 0 ---padding token, always zero-padding
            else 
                local tok = tokens[step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx[self.unkToken]
                end
                inputTensor[sId][step] = tok_id
            end
        end
    end
end

function SimpleBiLSTM:buildVocab(sentences, sentence_toks)
    if self.idx2word ~= nil then
        return 
    end
    local embeddingObject = self.embeddingObject
    local embW2V = embeddingObject.w2vvocab

    self.idx2word = {}
    self.word2idx = {}
    self.vocabSize = 0
    self.vocabSize = self.vocabSize + 1
    self.word2idx[self.unkToken] = self.vocabSize
    self.idx2word[self.vocabSize] = self.unkToken
    self.unkTokens = {}
    self:buildVocabForTokens(sentences, sentence_toks, embW2V)
    print("number of unique words (including unknown):".. self.vocabSize.." (unknown words are replaced by unk)")
    print("number of unknown tokens (not unique): ".. countTable(self.unkTokens))
end

function SimpleBiLSTM:buildVocabForTokens(sentences, toks, embW2V)
    for i = 1, #sentences do
        local tokens = toks[i]
        for j = 1, #tokens do 
            local tok = tokens[j]
            local tokInEmbId = embW2V[tok]
            if tokInEmbId == nil and self.word2idx[tok] == nil then
                ---not in the pretrained embedding, just use unk
                self.word2idx[tok] = self.word2idx[self.unkToken]
                if self.unkTokens[tok] == nil then
                    self.unkTokens[tok] = 1 --dummy value
                end
            else
                --in the pretraining embedding table
                local tok_id = self.word2idx[tok]
                if tok_id == nil then 
                    self.vocabSize = self.vocabSize + 1
                    self.word2idx[tok] = self.vocabSize
                    self.idx2word[self.vocabSize] = tok
                end
            end
        end
    end
end

function printTable(table)
    for a,b in pairs(table) do print(a,b) end
end

function countTable(table)
    local count = 0
    for _ in pairs(table) do count = count + 1 end
    return count
end

