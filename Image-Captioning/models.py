import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim      # 2048
        self.attention_dim = attention_dim  # 512
        self.embed_dim = embed_dim          # 512
        self.decoder_dim = decoder_dim      # 512
        self.vocab_size = vocab_size        # 52
        self.dropout = dropout              # 0.5

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim) = (32, 512)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim) = (32, 512)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # preparation
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # Flatten image
        # (batch_size, num_pixels, encoder_dim) = (32, 196, 2048)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        num_pixels = encoder_out.size(1) # 196

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # Sort input data by decreasing lengths; why? apparent below
        """
        caption_lengths 原本为(32, 1), 现在经过sorted变成 descending降序并dim=0降维, 现在就是一个1维32长的降序tensor
            e.g.tensor([23, 16, 16, 15, 14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10])
        sort_ind[i] 为caption_lengths[i]在原caption_lengths中的索引
            e.g.tensor([16, 24, 25, 10, 2, 15, 0, 1, 7, 21, 27, 4, 5, 6, 26, 31, 8, 9, 11, 12, 13, 14, 18, 19, 20, 23, 28, 29, 30, 3, 17, 22])
        至于为啥要排序道理也很简单, 因为每一个图片的caption长度不同, 想要并行处理就需要有序
        """
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        # 由于caption_lengths按降序排列, 将对应的 encoder_out 和 encoded_captions也对应排序, 以便对应batch
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # Initialize LSTM state
        # encoder_out: (32, 196, 2048)
        h, c = self.init_hidden_state(encoder_out)  # h, c = (batch_size, decoder_dim)

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1

        # decode_lengths是caption_lengths每个元素-1形成的list, size = (32)
        decode_lengths = (caption_lengths - 1).tolist()

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # Create tensors to hold word predicion scores and alphas
        # 先申明 predictions 和 alphas 两个三维矩阵, 便于后面赋值

        # predictions = (32, 22, 9490) 其中22是这个batch=32中decode_lengths中的最大值
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # alphas = (32, 22, 196)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # ***********************************************************************************************************
        # ***********************************************************************************************************
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        for t in range(max(decode_lengths)):
            # batch_size_t 是指当前 batch=32 里的 decode_lengths 中大于当前时间步数t的个数
            # e.g. 当前 t = 10, 则 t=10 的循环里 batch_size_t = 16, 意味着32个decode_lengths里有16个 > 10
            batch_size_t = sum([l > t for l in decode_lengths])
            
            """
            [:batch_size_t]取出了前 batch_size_t 个batch, 这些batch是仍然有caption可以处理的, 被筛掉的batch已经处理完成
            原来encoder_out = (32, 196, 2048)代表用2048维向量表示196个pixel
            现在attention_weighted_encoding = (32, 2048)

            如何正确理解这个很重要:
            encoder_out[:batch_size_t]相当于keys && values,
            h[:batch_size_t]相当于queries。
            此处做的是一个cross-attention,
            完成了encoder_out对于前面隐状态(即你的caption在t时间步前输出的那些letter)的注意力,
            实际上不难发现, lstm只能根据前面看到的作记忆, 这是与attention相比的弊端, 而这样正好完成了mask-attention
            """
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            
            """
            attention_weighted_encoding = (32, 2048), 表示每一个batch即每一张图片得到2048维向量特征
            gate 是通过上一个隐状态接linear得到, 代表了与前一个输出即train的caption中的前一个letter相关
            两者相乘得到了新的attention_weighted_encoding, 加强attention, 理解为一个小trick
            """
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding # (32, 2048)

            """
            embeddings = (32, 52, 512)
            embeddings[:batch_size_t, t, :] 表示 batch 取前 batch_size_t 个, 第二维取定 =t, 最后全取, 故变成(32, 512)
            **此处直接将embeddings即captions输入而不是上一个隐状态, 用到teacher forcing

            attention_weighted_encoding = (32, 2048), 表示根据 h 隐状态决定的下一个注意力在何处
            cat将两者连接, 输入到lstm当中, 更新h和c
            """
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            
            # 一个全连接层, 将 h 隐状态转化为 (32, 9490), 9490是vocabulary中9490个词对应的概率, 概率最大的那个就是本轮的输出
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            # 对已分配空间的predictions, alphas赋值
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
