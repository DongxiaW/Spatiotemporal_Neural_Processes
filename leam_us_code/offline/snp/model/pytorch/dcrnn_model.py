import numpy as np
import torch
import torch.nn as nn

from model.pytorch.dcrnn_cell import DCGRUCell

device = torch.device("cuda:2")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.num_rnn_layers_gru = int(model_kwargs.get('num_rnn_layers_gru', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input0_dim = int(model_kwargs.get('input0_dim', 24))
        self.context_percentage = float(model_kwargs.get('context_percentage', 0.5))


class EmbedModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])
        self.output_layer = nn.Linear(self.hidden_state_size, self.rnn_units)

    def forward(self, inputs, hidden_state=None):
        """
        DCRNN forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            # print('no starting hidden states')
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        output = self.output_layer(output) #county level to state level

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow



class NP_ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        self.in_dim = int(model_kwargs.get('r_dim')*2)
        self.out_dim = int(model_kwargs.get('z_dim'))
        
        self.m1 = torch.nn.Linear(self.in_dim, self.out_dim)
        self.var1 = torch.nn.Linear(self.in_dim, self.out_dim)

    def forward(self, inputs):
        mean = self.m1(inputs)
        var_temp = self.var1(inputs) #

        return mean, var_temp

class NP_init_ZEncoder(nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        self.in_dim = int(model_kwargs.get('r_dim'))
        self.out_dim = int(model_kwargs.get('z_dim'))
        
        self.m1 = torch.nn.Linear(self.in_dim, self.out_dim)
        self.var1 = torch.nn.Linear(self.in_dim, self.out_dim)
        
    def forward(self, inputs):
        mean = self.m1(inputs)
        var_temp = self.var1(inputs) #

        return mean, var_temp

class NP_REncoder(nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, rnn_enc_in_dim, rnn_enc_out_dim):
        nn.Module.__init__(self)
        self.in_dim = rnn_enc_in_dim
        self.out_dim = rnn_enc_out_dim
        self.output_layer1 = nn.Linear(self.in_dim, self.out_dim)
        self.output_layer2 = nn.Linear(self.out_dim, self.out_dim)
        self.output_layer3 = nn.Linear(self.out_dim, self.out_dim)
        
    def forward(self, inputs):
        # output = self.output_layer(inputs)
        return self.output_layer3(torch.sigmoid(self.output_layer2(torch.sigmoid(self.output_layer1(inputs)))))

class NP_Decoder(nn.Module):
    """Takes an r representation and produces the mean & standard deviation of the 
    normally distributed function encoding, z."""
    def __init__(self, rnn_dec_in_dim, rnn_dec_out_dim):
        nn.Module.__init__(self)
        self.in_dim = rnn_dec_in_dim
        self.out_dim = rnn_dec_out_dim
        self.output_layer1 = nn.Linear(self.in_dim, self.out_dim)
        self.output_layer2 = nn.Linear(self.out_dim, self.out_dim)
        self.output_layer3 = nn.Linear(self.out_dim, self.out_dim)
        
    def forward(self, inputs):
        # output = self.output_layer(inputs)
        return self.output_layer3(torch.sigmoid(self.output_layer2(torch.sigmoid(self.output_layer1(inputs)))))


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, latent_dim=50, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.embed_model = EmbedModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self.NP_z_init_encoder = NP_init_ZEncoder(**model_kwargs) # r-> mean, var_temp
        self.NP_z_encoder = NP_ZEncoder(**model_kwargs) # r-> mean, var_temp

        self.seq_len = int(model_kwargs.get('seq_len'))
        self.rnn_enc_in_dim = int(model_kwargs.get('rnn_units')) + int(model_kwargs.get('output_dim')*2)
        self.rnn_enc_out_dim = int(model_kwargs.get('r_dim'))
        self.NP_r_encoder = NP_REncoder(self.rnn_enc_in_dim, self.rnn_enc_out_dim)
        self.snp_encoder = nn.GRU(self.rnn_enc_out_dim*2, self.rnn_enc_out_dim, self.num_rnn_layers_gru)
        self.rnn_dec_in_dim = int(model_kwargs.get('rnn_units')) + int(model_kwargs.get('output_dim')) + int(model_kwargs.get('z_dim'))
        self.rnn_dec_out_dim = int(model_kwargs.get('rnn_units_gru'))
        self.snp_decoder = NP_Decoder(self.rnn_dec_in_dim, self.rnn_dec_out_dim)
        self.fc = nn.Linear(int(model_kwargs.get('rnn_units_gru')), int(model_kwargs.get('output_dim')))
        self.relu = nn.ReLU()
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def split_context_target(self,inputs, labels, inputs0, context_percentage):
        """Helper function to split randomly into context and target"""
        n_context = int(inputs.shape[1]*context_percentage)
        ind = np.arange(inputs.shape[1])
        mask = np.random.choice(ind, size=n_context, replace=False)
        others = np.delete(ind,mask)

        return inputs[:,mask], labels[:,mask], inputs0[mask], inputs[:,others], labels[:,others], inputs0[others]

    def data_to_r_params(self, start, x, y):
        """Helper to batch together some steps of the process."""
        outputs_hidden = self.dcrnn_to_hidden(x)
        prev_day_seq = torch.cat([torch.unsqueeze(start,dim=0),(y[:-1]).clone().detach()],dim=0)

        xy = torch.cat([outputs_hidden,prev_day_seq,y], dim=-1) #feature: 16+24+24
        rs = self.NP_r_encoder(xy) # representation of xy
        r_agg = torch.mean(rs, dim=1) # Average over samples
        return r_agg # Get mean and variance for q(z|...)
    
    def sample_z(self, mean, var, n=1):
        """Reparameterisation trick."""
        eps = torch.autograd.Variable(var.data.new(var.size(0),n,var.size(1)).normal_()).to(device)

        std = torch.sqrt(var)
        return torch.unsqueeze(mean, dim=1) + torch.unsqueeze(std, dim=1) * eps

    def dcrnn_to_hidden(self, x):
        """
        dcrnn forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        inputs = x
        outputs_hidden = []
        encoder_hidden_state = None
        for t in range(self.embed_model.seq_len):
            output_hidden, encoder_hidden_state = self.embed_model(inputs[t], encoder_hidden_state)
            outputs_hidden.append(output_hidden)

        return torch.stack(outputs_hidden,dim=0)

    def snp(self, r_inputs):

        # batch_size = int(1)
        r_inputs = torch.unsqueeze(r_inputs,dim=1)
        z_mean_list = []
        z_var_temp_list = []
        z_mean, z_var_temp = self.NP_z_init_encoder(r_inputs[:1])
        z_mean_list.append(z_mean)
        z_var_temp_list.append(z_var_temp)

        # hidden_state = torch.zeros((self.num_rnn_layers_gru, 1, self.rnn_enc_out_dim), device=device)
        hidden_state = None

        for t in range(self.seq_len-1):
            output_interal, hidden_state = self.snp_encoder(torch.cat([z_mean, z_var_temp],dim=-1), hidden_state)
            z_mean, z_var_temp = self.NP_z_encoder(torch.cat([output_interal, r_inputs[t+1:t+2]],dim=-1))
            z_mean_list.append(z_mean)
            z_var_temp_list.append(z_var_temp)

        z_mean_seq = torch.squeeze(torch.cat(z_mean_list,dim=0),dim=1)
        z_var_temp_seq = torch.squeeze(torch.cat(z_var_temp_list,dim=0),dim=1)

        return z_mean_seq, z_var_temp_seq


    def decoder(self, start, outputs_hidden, zs, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = outputs_hidden.size(1)
        decoder_inputs = outputs_hidden
        decoder_output = torch.unsqueeze(start,dim=0)

        outputs = []

        for t in range(self.seq_len):
            decoder_output_interal = self.snp_decoder(torch.cat([decoder_inputs[t:t+1],zs[t:t+1], decoder_output],dim=-1))
            decoder_output = self.fc(self.relu(decoder_output_interal))
            outputs.append(decoder_output)

        outputs = torch.cat(outputs,dim=0)

        return outputs

    def forward(self, inputs, labels, inputs0, batches_seen=None, test=False, z_mean_all=None, z_var_temp_all=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param inputs0: shape (batch_size, input_dim)
        :param labels: shape (horizon, batch_size, output)
        :param batches_seen: batches seen till now
        :param test: train or test
        :param z_mean_all: z_mean_all for the last training epoch
        :param z_var_all: z_var_all for the last training epoch
        :return: outputs,truth: (self.horizon, batch_size, self.output_dim)
                 z_mean_all, z_var_all, z_mean_context, z_var_context (self.horizon, batch_size, z_dim)
        """
        if test==False:

            self._logger.debug("starting point complete, starting split source and target")
            #first half for context, second for target
            x_c, y_c, start_c, x_t, y_t, start_t = self.split_context_target(inputs, labels, inputs0, self.context_percentage)
            self._logger.debug("data split complete, starting encoder")
            # h_ct_encoder = torch.unsqueeze(self.fc_startlayer(inputs0),dim=0).repeat_interleave(self.num_rnn_layers,dim=0)
            # h_c_encoder = torch.unsqueeze(self.fc_startlayer(start_c),dim=0).repeat_interleave(self.num_rnn_layers,dim=0)
            r_agg_all = self.data_to_r_params(inputs0, inputs, labels) #(seq, dim)
            z_mean_all, z_var_temp_all = self.snp(r_agg_all)
            z_var_all = 0.1+ 0.9*torch.sigmoid(z_var_temp_all)

            r_agg_context = self.data_to_r_params(start_c, x_c, y_c)
            z_mean_context, z_var_temp_context = self.snp(r_agg_context)
            z_var_context = 0.1+ 0.9*torch.sigmoid(z_var_temp_context)

            zs = self.sample_z(z_mean_all, z_var_all, y_t.size(1))
            self._logger.debug("Encoder complete, starting decoder")

            outputs_hidden = self.dcrnn_to_hidden(x_t)
            output = self.decoder(start_t, outputs_hidden, zs)
            truth = y_t

            self._logger.debug("Decoder complete")
            if batches_seen == 0:
                self._logger.info(
                    "Total trainable parameters {}".format(count_parameters(self))
                )
            return output, truth, z_mean_all, z_var_temp_all, z_mean_context, z_var_temp_context
            
        else:
            # h_t_decoder = torch.unsqueeze(self.fc_startlayer(inputs0),dim=0).repeat_interleave(self.num_rnn_layers,dim=0)
            z_var_all = 0.1+ 0.9*torch.sigmoid(z_var_temp_all)
            zs = self.sample_z(z_mean_all, z_var_all, labels.size(1))
            outputs_hidden = self.dcrnn_to_hidden(inputs)
            # xz = torch.cat([h_outputs, zs], dim=-1)
            output = self.decoder(inputs0, outputs_hidden, zs)
            truth = labels

            return output, truth

