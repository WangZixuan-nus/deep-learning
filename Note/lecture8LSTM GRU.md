some additional learning notes in lec8 to better understand the contents.
 
# LSTM(Long short-term memory) model 
长短时记忆网络（LSTM）通过引入多个门控结构来控制信息在序列中的流动，其核心结构主要包括输入门（input gate）、遗忘门（forget gate）、输出门（output gate）、输入节点（input node，或称候选状态）和内部状态（internal state，也叫细胞状态）。

1. 输入门（Input Gate）  
   输入门决定当前时刻的新输入信息中，有哪些部分应该写入到内部状态中。具体流程如下：  
   - 首先，利用上一时刻的隐藏状态（hₜ₋₁）和当前输入（xₜ）计算出一个信号，通过一个 sigmoid 激活函数，输出范围在 0 到 1 之间。这个输出称为输入门的门控值（iₜ），它决定对候选状态的“采纳”程度。  
   - 同时，模型会计算一个候选状态（C̃ₜ，即 input node），通常通过 tanh 激活函数，将上一时刻的隐藏状态与当前输入进行非线性变换，实现对新信息的编码。  
   - 最后，将输入门信号与候选状态相乘，得到的新信息就会被加入到内部状态中。  
   这样的设计保证了模型可以有选择地采纳新信息，并在一定程度上抑制噪声。

2. 遗忘门（Forget Gate）  
   遗忘门的作用是决定内部状态中哪些信息需要被遗忘或舍弃。具体做法为：  
   - 同样利用上一时刻的隐藏状态和当前输入，经过一个 sigmoid 激活函数生成一个遗忘门信号（fₜ），其取值范围也是 0 到 1。  
   - 将这个信号与上一时刻的内部状态（Cₜ₋₁）相乘，得到了经过“筛选”的内部状态部分，这部分信息会被保留；而接近 0 的部分则被“遗忘”掉。  
   这一过程使得 LSTM 可以清除不再重要的历史信息，从而在长序列中有效防止无用信息的积累。

3. 输出门（Output Gate）  
   输出门负责决定当前单元的输出隐藏状态（hₜ），即将内部状态的哪些部分传递作为输出。其工作流程为：  
   - 同样将上一时刻的隐藏状态与当前输入合起来，通过一个 sigmoid 激活函数生成输出门信号（oₜ）。  
   - 内部状态（Cₜ）经过 tanh 激活函数处理后，再与输出门信号相乘，形成最终的隐藏状态 hₜ。  
   输出门保证了模型在不同时间步能够灵活决定将内部状态中哪些信息用于后续计算或输出，帮助捕捉动态变化的时序模式。

4. 输入节点（Input Node / Candidate State）  
   输入节点也称为候选状态（C̃ₜ），它代表了当前时刻从输入和过去信息中“候选”出来的新信息。  
   - 计算方式上，模型会将上一时刻的隐藏状态和当前输入进行线性组合后，再经过 tanh 激活函数来产生一个新状态，其取值范围在 -1 到 1 之间。  
   - 这个候选状态反映了在当前时刻可能需要写入内部状态的新信息，但最终是否写入还要依赖输入门的控制信号。  
   通过这种方式，LSTM 能够引入新信息，同时与保留下来的历史信息协同更新内部状态。

5. 内部状态（Internal State / Cell State）  
   内部状态（通常记作 Cₜ）可以看作是 LSTM 的“记忆”，它在序列中横向传递，承担着长时间依赖信息的存储工作。  
   - 更新内部状态需要结合遗忘门和输入门的作用：  
     ① 遗忘门根据 fₜ 决定遗忘上一时刻的哪些信息；  
     ② 输入门决定将候选状态 C̃ₜ 中的新信息写入多少比例。  
   - 通过这种组合，内部状态不断更新和传递，使得 LSTM 能够捕捉并长期保留有用信息，同时剔除冗余、无用的信息。  
   内部状态就像一条高速公路，几乎可以不经干扰地将关键信息传递到后续时刻，从而解决了传统 RNN 中梯度消失的问题。

总体来说，这几个门控和节点协同工作，可以使 LSTM 在每个时间步动态选择保留、更新或输出信息，从而在处理长序列数据时既能捕捉短期变化，也能记住重要的长期依赖关系。这种机制使得 LSTM 成为了处理自然语言、语音和其他序列数据的非常强大的模型。


---
# GRU(Gated recurrent unit) model

GRU（Gated Recurrent Unit，门控循环单元）是一种常用的循环神经网络变体，由 Cho 等人在 2014 年提出。它与长短时记忆网络（LSTM）类似，都用于解决传统 RNN 在处理长序列数据时存在的梯度消失或梯度爆炸问题，但在结构上更加简单。下面详细讲解一下 GRU 的关键组成部分、工作原理以及它与 LSTM 的比较。

---

### 1. GRU 的组成部分

GRU 主要包含两个门控机制：

- **重置门（Reset Gate）**  
  重置门用于决定当前输入和之前的隐藏状态在生成候选隐藏状态时应丢弃多少旧信息。  
  - 数学上，重置门 rₜ 计算公式通常为：  
    rₜ = σ(W_r · [hₜ₋₁, xₜ] + b_r)  
  - 这里的 σ 表示 sigmoid 激活函数，hₜ₋₁为上一时刻的隐藏状态，xₜ为当前输入。  
  - 当 rₜ 接近 0 时，说明忽略之前的状态，从而使模型“重置”记忆；而当 rₜ 接近 1 时，之前的信息被较大程度保留。

- **更新门（Update Gate）**  
  更新门决定当前时刻的隐藏状态应保留多少之前的信息，同时引入多少新的候选状态信息。  
  - 数学上，更新门 zₜ 的计算公式为：  
    zₜ = σ(W_z · [hₜ₋₁, xₜ] + b_z)  
  - 更新门的作用类似于 LSTM 中的输入门与遗忘门的组合。  
  - zₜ 的输出范围在 0 到 1 之间，调控信息的流动和记忆保持。

此外，GRU 还会计算一个候选隐藏状态（candidate hidden state），记作 h̃ₜ，其计算方式结合了当前输入和被重置后的之前隐藏状态：
  
- 候选隐藏状态 h̃ₜ 的计算公式为：  
  h̃ₜ = tanh(W · [rₜ * hₜ₋₁, xₜ] + b)  
- 其中 rₜ * hₜ₋₁ 表示对上一时刻状态施加重置门后，再与当前输入结合进行非线性变换。

---

### 2. GRU 的状态更新

GRU 的最终隐藏状态 hₜ 是通过将上一时刻的隐藏状态 hₜ₋₁ 与新的候选隐藏状态 h̃ₜ 按照更新门 zₜ 得到的加权平均结果。具体公式如下：

  hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ

从上式可以看出：
- 当更新门 zₜ 接近 0 时，当前隐藏状态主要保留上一时刻的隐藏状态，即保留历史记忆。
- 当更新门 zₜ 接近 1 时，当前隐藏状态则更多地采用新的候选状态信息，从而实现信息更新。

---

### 3. GRU 与 LSTM 的比较

- **结构简化**  
  GRU 只有两个门（重置门和更新门），而 LSTM 则有三个门（遗忘门、输入门和输出门）。因此，GRU 的结构相对更简单，计算量和参数量也较少。
  
- **记忆能力**  
  虽然结构更简洁，但在很多实际任务中，GRU 的性能可以与 LSTM 相媲美，甚至在某些场景下表现更好。两者各有优势，具体表现依赖于任务和数据集。
  
- **训练速度**  
  由于 GRU 结构简单，参数较少，因此在训练速度上往往会比 LSTM 更快，尤其在数据量较大或设备资源有限的情况下表现优异。

---

### 4. GRU 的代码示例

下面提供一个简单的 PyTorch 实现 GRU 单元的示例代码。

````python name=gru_example.py
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        # 更新门：计算 z_t
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 重置门：计算 r_t
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # 候选隐藏状态：计算 h̃_t
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # 将输入和上一时刻隐藏状态拼接
        combined = torch.cat([h_prev, x], dim=1)
        # 计算更新门
        z_t = torch.sigmoid(self.update_gate(combined))
        # 计算重置门
        r_t = torch.sigmoid(self.reset_gate(combined))
        # 对上一时刻隐藏状态施加重置门
        combined_candidate = torch.cat([r_t * h_prev, x], dim=1)
        # 计算候选隐藏状态
        h_candidate = torch.tanh(self.candidate(combined_candidate))
        # 更新隐藏状态
        h_t = (1 - z_t) * h_prev + z_t * h_candidate
        return h_t

# 示例使用
if __name__ == "__main__":
    input_size = 5   # 输入维度
    hidden_size = 3  # 隐藏状态维度
    gru_cell = GRUCell(input_size, hidden_size)
    
    # 模拟一个时间步的输入，批大小为1
    x = torch.randn(1, input_size)
    h_prev = torch.zeros(1, hidden_size)
    
    h = gru_cell(x, h_prev)
    print("更新后的隐藏状态 h:", h)
````

---

### 总结

GRU 通过重置门和更新门机制简化了信息流控制，相比 LSTM 更加简单高效，适用于大部分序列建模任务。其核心思想在于利用更新门来平衡旧记忆和新信息的比例，而重置门则决定了信息融合时对历史状态的缩放。通过这些机制，GRU 能够在不牺牲表达能力的前提下，降低模型复杂度并提速训练。


---
深度学习中的 Encoder（编码器）和 Decoder（解码器）结构最早在序列到序列（Seq2Seq）模型中得到广泛应用，后来又扩展到机器翻译、文本摘要、图像描述等多种任务中。下面详细讲解这两个部分的原理、功能以及它们如何协同工作。

---
### 5. Encoder-decoder for machine translation
#### 1. Encoder（编码器）

##### 功能
编码器的主要任务是将输入数据（例如，一个句子或一张图像）转换为一个内部的、高维的、抽象的表示，这个表示通常能够捕捉输入数据中的关键信息。对于序列数据，编码器会对序列中的每个元素进行逐步处理，并产生一个“隐藏状态”序列，还可能生成一个全局的状态（如最后一个隐藏状态或经过拼接、池化后的表示）。

#### 常见结构
- **RNN/LSTM/GRU 编码器**  
  对于文本或语音等时序数据，经常使用循环神经网络（RNN）或它的改进版本（LSTM、GRU）构建编码器。输入序列经过网络逐步处理，每个时间步都会输出一个隐藏状态，最终的隐藏状态可以看作整个序列的摘要。
  
- **卷积神经网络（CNN）编码器**  
  在图像处理任务中，通常使用 CNN 作为编码器，通过多层卷积、池化操作将原始图像抽象成特征映射，从而捕获局部和全局的图像特征。

- **Transformer Encoder**  
  近年来，Transformer 编码器以自注意力机制为核心，不需要递归就能捕捉序列中所有位置之间的依赖关系，这使得编码器在处理长序列时更高效。Transformer 编码器由多个自注意力模块和前馈网络层堆叠而成，每一层都会生成一组表示向量。

##### Encoder Outputs 和 Encoder State
- **Encoder Outputs**  
  指的是编码器对输入序列中每个时刻产生的隐藏状态向量序列。这些输出通常携带输入中局部信息，并可以用于后续注意力机制的计算，帮助 Decoder 在生成时关注相关部分信息。
  
- **Encoder State**  
  通常是编码器处理整个输入序列后得到的全局状态（如最后一个隐藏状态），它概括了整个输入序列的信息。这一状态常用于初始化 Decoder 的状态，使解码器从一开始就能融入输入数据的整体信息。

---

#### 2. Decoder（解码器）

##### 功能
解码器的作用是根据 encoder 提取的输入特征以及时刻变化的上下文信息（有时用到 Encoder Outputs 与 Encoder State）逐步生成输出数据（例如另一种语言的翻译结果、摘要或者下一个可能的单词）。

##### 常见结构
- **RNN/LSTM/GRU 解码器**  
  最早的 Seq2Seq 模型中，解码器也是基于循环神经网络构造，接收编码器的全局状态作为初始状态，并在生成每个输出时结合先前生成的结果不断更新状态。激活时通常还会结合注意力机制来利用 encoder outputs 得到上下文向量。
  
- **Transformer Decoder**  
  在 Transformer 架构中，解码器同样由多层自注意力和交叉注意力模块构成。自注意力模块用于捕捉解码器内部生成的序列依赖，而交叉注意力模块则通过对 encoder outputs 的查询，确保生成时参考输入数据。

##### 注意力机制与上下文变量（Context Variable）
- 由于简单地使用 encoder 的最终状态往往无法捕获输入中所有细粒度的信息，很多模型在解码过程中引入注意力机制。通过计算解码器当前状态与 encoder outputs 中每个时间步的相似度（或“对齐”得分），可以得到一组权重，再对 encoder outputs 进行加权求和，得到的向量称为上下文向量或 context variable。
- 这个上下文向量会与解码器的当前状态共同决定下一个输出，这使得解码器在生成过程中能够“关注”输入中与当前生成最相关的部分，从而提高输出的准确性和一致性。

---

#### 3. Encoder-Decoder 协同工作的流程

可以将整个过程大致分为以下几个步骤：
1. **编码阶段**
   - 输入数据（如源语言句子）送入 Encoder，产生一组隐藏状态（Encoder Outputs）和/或一个全局状态（Encoder State）。
  
2. **初始化解码器**
   - 将 Encoder State 作为解码器的初始状态，为生成输出提供整体上下文。
  
3. **解码阶段**
   - 解码器开始逐步生成输出（如目标语言句子），在每一步利用现在的状态和先前生成的输出，通过注意力机制计算上下文变量（Context Variable），从而动态地结合输入信息生成更合理的输出。
  
4. **生成输出**
   - 每一步生成的结果可能经过 Softmax 层转换为概率分布，选择概率最高的词作为下一个输出，重复这个过程直到产生结束标记。

---

#### 4. 示例代码（基于 PyTorch 的简单 Seq2Seq 结构）

以下代码示例展示了一个简化的 Encoder-Decoder 框架，其中包含一个基于 LSTM 的编码器和解码器，同时在解码器中加入了注意力机制来生成上下文变量。

```python name=seq2seq_example.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 编码器：基于 LSTM
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src)  # [batch_size, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim]
        # hidden, cell: [n_layers, batch_size, hid_dim]
        return outputs, hidden, cell

# 简单的注意力机制
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim] 当前解码器隐藏状态
        # encoder_outputs: [batch_size, src_len, hid_dim]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 将当前隐藏状态扩展到与 encoder 输出相同的长度
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        energy = energy.permute(0, 2, 1)  # [batch_size, hid_dim, src_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hid_dim]
        attention = torch.bmm(v, energy).squeeze(1)  # [batch_size, src_len]
        return F.softmax(attention, dim=1)

# 解码器：基于 LSTM，带注意力机制
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.attention = attention
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size] 当前时刻输入（通常是上一步生成的词）
        # hidden, cell: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]
        
        # 计算注意力权重和上下文向量
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim]
        
        # 将上下文向量与嵌入向量拼接后输入到 RNN 中
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # output: [batch_size, 1, hid_dim]
        # 计算预测
        output = output.squeeze(1)  # [batch_size, hid_dim]
        context = context.squeeze(1)  # [batch_size, hid_dim]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch_size, output_dim]
        return prediction, hidden, cell

# 示例如何构造整个 Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        # 这里仅作为示例：进行编码，然后在解码阶段逐步生成输出
        encoder_outputs, hidden, cell = self.encoder(src)
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(src.device)
        # 通常第一个输入为 <sos>
        input = trg[:, 0]
        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = prediction
            # 在训练时使用 teacher forcing，可以替换下面这一句
            input = prediction.argmax(1)
        return outputs

# 构造模型示例参数
INPUT_DIM = 10000  # 源语言词汇表大小
OUTPUT_DIM = 10000  # 目标语言词汇表大小
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS)
attn = Attention(HID_DIM)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, attn)
model = Seq2Seq(enc, dec)

print(model)
```

---

#### 总结

- **Encoder（编码器）** 负责把输入数据转换为一个内部表示，可包括局部（每个时间步的输出）和全局信息（最终隐藏状态）。
- **Decoder（解码器）** 基于编码器的输出，逐步生成目标输出，并通过注意力机制利用输入序列中的信息（上下文变量）。
- 两者结合的结构使得模型能够灵活地捕获输入和输出之间的依赖关系，在机器翻译和其它生成任务中表现出色。
---
贪婪搜索（Greedy Search）、穷举搜索（Exhaustive Search）和束搜索（Beam Search）是深度学习中生成序列任务（如机器翻译、文本生成等）常用的三种策略，它们在生成解码器输出时的搜索方式各有不同，下面详细讲解它们的原理、优缺点及适用场景。

---
### 6. Prediction
#### 1. 穷举搜索（Exhaustive Search）

**原理：**  
- 穷举搜索是对所有可能的输出序列进行完整枚举；也就是说，在生成过程中，它会列举出所有可能的词序列组合，然后根据整体的得分（通常是各时间步概率的乘积或对数和）选出得分最高的序列作为最终输出。

**优点：**  
- 可以保证全局最优：在理论上，如果穷举所有可能的序列，就能找到令整体概率最高的解。

**缺点：**  
- 计算代价极高：输出序列的可能性组合数目呈指数增长，对于大词汇量和长序列来说，计算时间和资源无法接受；
- 实际应用中几乎不可能实现，除非目标序列非常短或者候选空间非常有限。

**适用场景：**  
- 通常作为理论分析基准或在非常小规模的生成任务中使用；在实际深度学习应用中基本不会采用穷举搜索。

---
#### 2. 贪婪搜索（Greedy Search）

**原理：**  
- 在每一步生成中，模型会对所有可能的候选输出计算出一个概率分布（通常经过 softmax 激活）。
- 贪婪搜索在每一个时间步仅选择当前概率最高的候选项，作为下一个输入，依次生成完整序列。

**优点：**  
- 计算速度快，过程简单高效；
- 实时性较好，适合对速度要求较高的场景。

**缺点：**  
- 每步只做局部最优选择，无法保证生成整个序列时达到全局最优；
- 可能会陷入局部最优解，忽略后续潜在更好的组合，生成的结果多为单一和缺少多样性。

**适用场景：**  
- 用于快速生成输出、基线系统的搭建或对生成质量要求不特别严格的应用中。

**简单示例：**

```python name=greedy_search_example.py
import torch
import torch.nn.functional as F

# 模拟一个可能的模型输出的 logits
logits = torch.tensor([2.0, 1.0, 0.5, 0.2])
# 计算概率分布
probabilities = F.softmax(logits, dim=0)
# 贪婪选择概率最大的下一个词
_, predicted_index = torch.max(probabilities, dim=0)
print("概率分布:", probabilities)
print("当前时间步选择的索引:", predicted_index.item())
```


---

#### 3. 束搜索（Beam Search）

**原理：**  
- 束搜索是介于贪婪搜索和穷举搜索之间的一种折中策略。
- 它在每个时间步不是只保留一个最优选项，而是保留固定数量（称为 beam width 或 beam size）的候选序列。
- 在下一步生成时，对于每个候选序列，模型都会扩展它们所有可能的后续输出，计算整个候选序列的累计得分，然后从所有扩展后的序列中再次选出 top-k 个效果最好的候选继续保留。

**优点：**  
- 通过同时跟踪多个候选序列，束搜索能更好地探索搜索空间，提高了生成全局最优解的概率；
- 在不显著增加复杂度的情况下，通常能生成比贪婪搜索更高质量、更自然的结果。

**缺点：**  
- 相比贪婪搜索，计算量和内存占用较高，因为需要维护多个候选解；
- 如果 beam size 设置过小，效果可能接近贪婪搜索；如果设置过大，则计算开销会显著增加且可能引入噪音。

**适用场景：**  
- 在机器翻译、文本生成、语音识别等任务中广泛使用；
- 尤其适用于对生成质量要求较高的场景，能在搜索全局最优和计算资源之间取得平衡。

**简单示例：**

下面伪代码展示了束搜索的基本逻辑：

```python name=beam_search_example.py
def beam_search(decoder, initial_state, beam_size, max_steps):
    # 初始状态含有 beam_size 个候选序列（初始时仅有一个候选序列）
    sequences = [(initial_state, [], 0.0)]  # 每个元素: (state, generated_sequence, cumulative_log_prob)
    for step in range(max_steps):
        all_candidates = []
        for state, seq, cum_log_prob in sequences:
            # 每个候选序列扩展所有可能后继词
            next_probs = decoder(state)  # 得到当前状态下每个词的概率分布（对数概率）
            for index, log_prob in enumerate(next_probs):
                new_seq = seq + [index]
                new_log_prob = cum_log_prob + log_prob
                new_state = update_state(state, index)
                all_candidates.append((new_state, new_seq, new_log_prob))
        # 按累计对数概率从高到低排序，保留前 beam_size 个候选序列
        sequences = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)[:beam_size]
    # 返回得分最高的候选序列
    return max(sequences, key=lambda tup: tup[2])
```

以上伪代码中，`decoder` 表示生成下一词的模块，`update_state` 表示根据当前选中的词更新隐藏状态的函数，`beam_size` 则是束的大小。

---

#### 总结

- **穷举搜索（Exhaustive Search）**：枚举所有可能的序列，能找到全局最优解，但计算代价巨大，实际应用中很少用。
- - **贪婪搜索（Greedy Search）**：每一步只做局部最优选择，速度快但不一定全局最优。
- **束搜索（Beam Search）**：在搜索过程中同时保留多个候选序列，通过权衡计算量和搜索深度，能生成更高质量的输出，是序列生成任务中常用的一种解码策略。

这些策略在深度学习中的序列生成任务中各有取舍，通常根据任务需求和计算资源选择合适的搜索方式。