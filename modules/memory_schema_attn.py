import torch
from torch.nn import Parameter as Param

class MemAttn(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 nhop):
        #
        # Input
        #  embedding_dim : number of embedding dimentions in memory
        #  nhop: number of memory hops
        #
        super(MemAttn, self).__init__()
        self.nhop = nhop

        self.tanh = torch.nn.Tanh()
        self.hidden = torch.nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor):
        #
        # Input
        #  Q : Matrix of queries: (batch_size, num_question_toks, embedding_dim)
        #  K : Matrix of keys: (batch_size, num_memories, embedding_dim)
        #  V : Matrix of values: (batch_size, num_memories, embedding_dim)
        #
        # Output
        #  R : soft-retrieval of values: (batch_size, num_question_toks, embedding_dim)
        #  attn_weights : soft-retrieval of values: (batch_size, num_question_toks, num_memories)
        Q_hop = Q
        for h in range(self.nhop):
            query_dim = torch.tensor(float(Q_hop.size(2)))
            if Q_hop.is_cuda: query_dim = query_dim.cuda()
            # (batch_size, num_question_toks, num_memories)
            attn_weights = torch.bmm(Q_hop,K.transpose(1,2))
            attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
            # (batch_size, num_question_toks, num_memories)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=2)
            # (batch_size, num_question_toks, embedding_dim)
            R = torch.bmm(attn_weights, V)
            # (batch_size, num_question_toks, embedding_dim*2)
            concat_by_step = torch.cat((Q_hop, R), 2)
            # (batch_size, num_question_toks, embedding_dim)
            Q_hop = self.tanh(self.hidden(concat_by_step))

        return Q_hop, attn_weights

    # def forward(self,
    #             final_encoder_out:torch.Tensor,
    #             schema: Dict[str, torch.LongTensor]):
    #     #
    #     # Input
    #     #  final_encoder_out : (batch_size, encoder_output_dim)
    #     #  schema: KnowledgeGraphField
    #     #
    #     schema_text = schema['text']
    #     embedded_schema_A = self.A(schema_text, num_wrapping_dims=1)
    #     embedded_schema_B = self.B(schema_text, num_wrapping_dims=1)
    #     schema_mask = util.get_text_field_mask(schema_text, num_wrapping_dims=1).float()
    #
    #     # (batch_size, num_entities, embedding_dim)
    #     Ain = self._entity_encoder_A(embedded_schema_A, schema_mask)
    #     Bin = self._entity_encoder_B(embedded_schema_B, schema_mask)
    #
    #     for h in range(self.nhop):
    #         # (batch_size, num_entities)
    #         Aout = torch.bmm(Ain, torch.transpose(final_encoder_out.unsqueeze(1), 1, 2)).squeeze(2)
    #         P = torch.nn.functional.softmax(Aout, dim=1)
    #
    #         # (batch, 1, embedding_dim)
    #         Bout = util.weighted_sum(P.squeeze(1), Bin)
    #
    #         Cout = torch.bmm(final_encoder_out, self.C)
    #         Dout = torch.cat(Cout, Bout)
    #     #
    #     # for h in range(self.nhop):
    #     #     self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
    #     #     Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True)
    #     #     Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
    #     #     P = tf.nn.softmax(Aout2dim)
    #     #
    #     #     probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
    #     #     Bout = tf.matmul(probs3dim, Bin)
    #     #     Bout2dim = tf.reshape(Bout, [-1, self.edim])
    #     #
    #     #     Cout = tf.matmul(self.hid[-1], self.C)
    #     #     Dout = tf.add(Cout, Bout2dim)
    #     #
    #     #     self.share_list[0].append(Cout)
    #     #
    #     #     if self.lindim == self.edim:
    #     #         self.hid.append(Dout)
    #     #     elif self.lindim == 0:
    #     #         self.hid.append(tf.nn.relu(Dout))
    #     #     else:
    #     #         F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
    #     #         G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim - self.lindim])
    #     #         K = tf.nn.relu(G)
    #     #         self.hid.append(tf.concat(axis=1, values=[F, K]))
    #
    #     return Dout

