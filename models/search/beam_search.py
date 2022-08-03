# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------

import torch
import utils
from einops import rearrange
import einops as eins


class BeamSearch(object):

    def __init__(self, model, max_len: int, eos_idx: int, beam_size: int):
        self.model = model
        self.max_len = max_len
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.b_s = None
        self.device = None
        self.seq_mask = None
        self.seq_logprob = None
        self.outputs = None
        self.log_probs = None
        self.selected_words = None
        self.all_log_probs = None

    def _expand_state(self, selected_beam, cur_beam_size):

        def fn(tensor):
            shape = [int(sh) for sh in tensor.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            tensor = torch.gather(tensor.view(*([self.b_s, cur_beam_size] + shape[1:])), 1,
                                  beam.expand(*([self.b_s, self.beam_size] + shape[1:])))
            tensor = tensor.view(*([-1] + shape[1:]))
            return tensor

        return fn

    def apply(self, visual, out_size=1, return_probs=False, **kwargs):
        if isinstance(visual, dict):
            self.b_s = visual['vis_feat'].shape[0]
            self.device = visual['vis_feat'].device
        else:
            self.b_s = visual.tensors.shape[0]
            self.device = visual.tensors.device

        # the mask of the current word (whether it != eos or not), it = 1 if != <eos>
        self.seq_mask = torch.ones((self.b_s, self.beam_size, 1), device=self.device)

        # the cummulative sum of log probs up to the current word [B, Beam, 1]
        self.seq_logprob = torch.zeros((self.b_s, 1, 1), device=self.device)

        # log probs of all beam_size selected words: [[B, Beam, 1] * max_len]
        self.log_probs = []
        self.selected_words = None
        if return_probs:
            self.all_log_probs = []

        # selected words at each timestep: [[B, Beam, 1] * max_len]
        outputs = []
        with self.model.statefulness(self.b_s):
            for t in range(self.max_len):
                visual, outputs = self.iter(t, visual, outputs, return_probs, **kwargs)

        # Sort result
        seq_logprob, sort_idxs = torch.sort(self.seq_logprob, 1, descending=True)

        # sum log_probs = seq_logprob
        # outputs = log_probs shape = [B, Beam, Len], the following is to sorted the order of which sequence.
        outputs = torch.cat(outputs, -1)  # [B, Beam, Len]
        outputs = torch.gather(outputs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        log_probs = torch.cat(self.log_probs, -1)
        log_probs = torch.gather(log_probs, 1, sort_idxs.expand(self.b_s, self.beam_size, self.max_len))
        if return_probs:
            all_log_probs = torch.cat(self.all_log_probs, 2)
            all_log_probs = torch.gather(
                all_log_probs, 1,
                sort_idxs.unsqueeze(-1).expand(self.b_s, self.beam_size, self.max_len, all_log_probs.shape[-1]))

        outputs = outputs.contiguous()[:, :out_size]  # [B Beam Len] -> [B, :topk, Len] select only the top k sentences
        log_probs = log_probs.contiguous()[:, :out_size]  # [B Beam Len] -> [B Len] select only the top k sentences
        if out_size == 1:
            outputs = outputs.squeeze(1)  # [B :topk, len] = [B, len] if topk = 1
            log_probs = log_probs.squeeze(1)

        if return_probs:
            return outputs, log_probs, all_log_probs
        else:
            return outputs, log_probs

    def select(self, t, candidate_logprob, **kwargs):
        candidate_logprob = rearrange(candidate_logprob, 'B Beam V -> B (Beam V)')
        selected_logprob, selected_idx = torch.sort(candidate_logprob, -1, descending=True)
        selected_logprob, selected_idx = selected_logprob[:, :self.beam_size], selected_idx[:, :self.beam_size]
        return selected_idx, selected_logprob  # [B Beam]

    def iter(self, t, samples, outputs, return_probs, **kwargs):
        cur_beam_size = 1 if t == 0 else self.beam_size

        word_logprob = self.model.step(t, self.selected_words, samples, None, mode='feedback', **kwargs)
        word_logprob = word_logprob.view(self.b_s, cur_beam_size, -1)  # [BB V] -> [B Beam V]
        candidate_logprob = self.seq_logprob + word_logprob  # [B Beam V]
        # Mask sequence if it reaches EOS
        if t > 0:
            _selected_words = self.selected_words.view(self.b_s, cur_beam_size)  # [BB, 1] -> [B Beam]
            # mask = 0 if it is eos, else 1.
            mask = eins.repeat((_selected_words != self.eos_idx).float(), 'B Beam -> B Beam V', V=1)
            self.seq_mask = self.seq_mask * mask  # [B Beam V] V=1
            word_logprob = word_logprob * self.seq_mask  # [B Beam V]
            old_seq_logprob = self.seq_logprob.expand_as(candidate_logprob).contiguous()
            old_seq_logprob[:, :, 1:] = -999  # [B Beam V]
            candidate_logprob = self.seq_mask * candidate_logprob + old_seq_logprob * (1 - self.seq_mask)  # [B Beam V]
            # Refer to line 269 field.py ['<unk>', '<pad>', '<bos>', '<eos>', 'a', 'on', ...] => old_seq_logprob[0] = '<unk>'
            # After <EOS>, we want to make all predictions to <UNK>.
            # When decoding, we will remove all predictions after <EOS>

        selected_idx, selected_logprob = self.select(t, candidate_logprob, **kwargs)
        selected_beam = selected_idx // candidate_logprob.shape[-1]  # [B Beam]
        selected_words = selected_idx - selected_beam * candidate_logprob.shape[-1]  # [B Beam]

        # save the states of the selected beam
        self.model.apply_to_states(self._expand_state(selected_beam, cur_beam_size))  # [BB, ...]

        self.seq_logprob = eins.repeat(selected_logprob, 'B Beam -> B Beam L', L=1)
        beam_exp = eins.repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.seq_mask = torch.gather(self.seq_mask, 1, beam_exp)
        outputs = [torch.gather(o, 1, beam_exp) for o in outputs]
        outputs.append(eins.repeat(selected_words, 'B Beam -> B Beam L', L=1))

        if return_probs:
            if t == 0:
                # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.expand((self.b_s, self.beam_size, -1)).unsqueeze(2))
            else:  # [B Beam V] -> [B Beam 1 V]
                self.all_log_probs.append(word_logprob.unsqueeze(2))

        beam_exp = eins.repeat(selected_beam, 'B Beam -> B Beam V', V=word_logprob.shape[-1])
        this_word_logprob = torch.gather(word_logprob, 1, beam_exp)
        this_word_logprob = torch.gather(this_word_logprob, 2, selected_words.unsqueeze(-1))

        beam_exp = eins.repeat(selected_beam, 'B Beam -> B Beam L', L=1)
        self.log_probs = [torch.gather(o, 1, beam_exp) for o in self.log_probs]

        self.log_probs.append(this_word_logprob)
        self.selected_words = selected_words.view(-1, 1)  # [B*Beam, 1]

        return samples, outputs
