import torch
from torch import nn
from engine.utils import NestedTensor
from einops import rearrange, repeat


class Ensemble(nn.Module):

    def __init__(
        self,
        models,
        bos_idx=2,
        use_vis_feat=True,
        use_reg_feat=False,
        config=None,
    ):
        super().__init__()
        self.models = models
        self.bos_idx = bos_idx
        self.use_reg_feat = use_reg_feat
        self.use_vis_feat = use_vis_feat
        self.config = config

    def get_bs_device(self, samples):
        if isinstance(samples, dict):
            batch_size = samples['gri_feat'].shape[0]
            device = samples['gri_feat'].device
        elif isinstance(samples, NestedTensor):
            batch_size = samples.tensors.shape[0]
            device = samples.tensors.device
        return batch_size, device

    def init_state(self, batch_size, device):
        return [torch.zeros((batch_size, 0), dtype=torch.long, device=device), None, None]

    def select(self, t, candidate_logprob, beam_size, **kwargs):
        candidate_logprob = rearrange(candidate_logprob, 'B Beam V -> B (Beam V)')
        selected_logprob, selected_idx = torch.sort(candidate_logprob, -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :beam_size], selected_idx[:, :beam_size]
        return selected_idx, selected_logprob  # [B Beam]

    def _expand_state(self, selected_beam, cur_beam_size, batch_size, beam_size):

        def fn(tensor):
            shape = [int(sh) for sh in tensor.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            tensor = torch.gather(tensor.view(*([batch_size, cur_beam_size] + shape[1:])), 1,
                                  beam.expand(*([batch_size, beam_size] + shape[1:])))
            tensor = tensor.view(*([-1] + shape[1:]))
            return tensor

        return fn

    def forward(self,
                samples,
                seq,
                use_beam_search=False,
                max_len=20,
                eos_idx=3,
                beam_size=5,
                out_size=1,
                return_probs=False,
                **kwargs):
        batch_size, device = self.get_bs_device(samples)

        # the mask of the current word (whether it != eos or not), it = 1 if != <eos>
        self.seq_mask = torch.ones((batch_size, beam_size, 1), device=device)

        # the cummulative sum of log probs up to the current word [B, Beam, 1]
        self.seq_logprob = torch.zeros((batch_size, 1, 1), device=device)

        # log probs of all beam_size selected words: [[B, Beam, 1] * max_len]
        self.log_probs = []
        self.selected_words = None

        if return_probs:
            self.all_log_probs = []

        # selected words at each timestep: [[B, Beam, 1] * max_len]
        outputs = []

        for model in self.models:
            model.enable_statefulness(batch_size)

        for timestep in range(max_len):
            samples, outputs = self.iter(
                timestep=timestep,
                samples=samples,
                outputs=outputs,
                return_probs=return_probs,
                batch_size=batch_size,
                beam_size=beam_size,
                eos_idx=eos_idx,
                **kwargs,
            )

        for model in self.models:
            model.disable_statefulness()

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)

        # sum log_probs = seq_logprob
        # outputs = log_probs shape = [B, Beam, Len], the following is to sorted the order of which sequence.
        outputs = torch.cat(outputs, -1)  # [B, Beam, Len]
        outputs = torch.gather(outputs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(
                all_log_probs, 1,
                sort_idxs.unsqueeze(-1).expand(batch_size, beam_size, max_len, all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]  # [B Beam Len] -> [B, :topk, Len] select only the top k sentences
        log_probs = log_probs.contiguous()[:, :out_size]  # [B Beam Len] -> [B Len] select only the top k sentences
        if out_size == 1:
            outputs = outputs.squeeze(1)  # [B :topk, len] = [B, len] if topk = 1
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def step(self, timestep, prev_output, samples, seq, mode='teacher_forcing', **kwargs):
        it = None
        step_outputs = []
        for model in self.models:
            if timestep == 0:
                if not model.cached_features:
                    vis_inputs = model.detector(samples)
                else:
                    vis_inputs = samples
                model.gri_feat, model.gri_mask = model.encoder(vis_inputs['gri_feat'], vis_inputs['gri_mask'])
                if model.config.model.decoder.decoder_name != 'mesh':
                    model.gri_feat = model.gri_feat[:, -1]

                if model.config.model.use_reg_feat:
                    model.det_feat = vis_inputs['det_feat']
                    model.reg_mask = vis_inputs['reg_mask']

                # If t = 0, enc_output = [B, N, D], init_tokens = [B, 1]
                # Else t > 0, enc_output = [BB, N, D], it = prev_output (t-1) = [BB, 1]
                it = model.gri_feat.data.new_full((model.gri_feat.shape[0], 1), model.bos_idx).long()
            else:
                it = prev_output

            vis_inputs = {
                'gri_feat': model.gri_feat,
                'gri_mask': model.gri_mask,
            }
            if model.config.model.use_reg_feat:
                vis_inputs['det_feat'] = model.det_feat
                vis_inputs['reg_mask'] = model.reg_mask
            step_outputs.append(model.decoder(it, vis_inputs))

        return torch.mean(torch.stack(step_outputs), dim=0)

    def iter(self, timestep, samples, outputs, return_probs, batch_size, beam_size=5, eos_idx=3, **kwargs):
        cur_beam_size = 1 if timestep == 0 else beam_size

        word_logprob = self.step(timestep, self.selected_words, samples, None, mode='feedback', **kwargs)
        word_logprob = word_logprob.view(batch_size, cur_beam_size, -1)  # [BB V] -> [B Beam V]
        candidate_logprob = self.seq_logprob + word_logprob  # [B Beam V]

        # Mask sequence if it reaches EOS
        if timestep > 0:
            _selected_words = self.selected_words.view(batch_size, cur_beam_size)  # [BB, 1] -> [B Beam]
            # mask = 0 if it is eos, else 1.
            mask = repeat((_selected_words != eos_idx).float(), 'B Beam -> B Beam V', V=1)
            self.seq_mask = self.seq_mask * mask  # [B Beam V] V=1
            word_logprob = word_logprob * self.seq_mask  # [B Beam V]
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999  # [B Beam V]
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)  # [B Beam V]
            # Refer to line 269 field.py ['<unk>', '<pad>', '<bos>', '<eos>', 'a', 'on', ...] => old_seq_logprob[0] = '<unk>'
            # After <EOS>, we want to make all predictions to <UNK>.
            # When decoding, we will remove all predictions after <EOS>

        selected_idx, selected_logprob = self.select(timestep, candidate_logprob, beam_size, **kwargs)
        selected_beam = torch.div(selected_idx, candidate_logprob.shape[-1], rounding_mode='floor')  # [B Beam]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]  # [B Beam]

        # save the states of the selected beam
        for model in self.models:
            model.apply_to_states(self._expand_state(selected_beam, cur_beam_size, batch_size, beam_size))  # [BB, ...]

        self.seq_logprob = repeat(selected_logprob, 'B Beam -> B Beam L', L=1)
        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.seq_mask = torch.gather(self.seq_mask, 1, beam_exp)
        outputs = [torch.gather(o, 1, beam_exp) for o in outputs]
        outputs.append(repeat(selected_words, 'B Beam -> B Beam L', L=1))

        if return_probs:
            if timestep == 0:
                # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.expand((batch_size, beam_size, -1)).unsqueeze(2))
            else:  # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob.shape[-1])
        this_word_logprob = torch.gather(word_logprob, 1, beam_exp)
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

        beam_exp = repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.log_probs = [torch.gather(o, 1, beam_exp) for o in self.log_probs]

        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)  # [B*Beam, 1]

        return samples, outputs
