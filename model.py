"""ggml-python implemention of the xtts model
"""
import io
import os
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple, Dict
import ggml

from ggml import ggml_new_tensor_1d, ggml_new_tensor_2d, GGML_TYPE_F32, ggml_row_size, ggml_init_params, ggml_init

def compute_ctx_size(n_embd, n_layer, n_ctx, n_mel_vocab, n_text_vocab, n_mel_position, n_text_position):
    ctx_size = 0
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_text_vocab * n_embd); # text embedding weights
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_text_position * n_embd); # text position embedding weights
    ctx_size += ggml_row_size(GGML_TYPE_F32, 32 * n_embd); # conditioning latent
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_mel_vocab * n_embd); # mel embedding weight
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_mel_position * n_embd); # mel position embedding weight

    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # ln_f_g
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # ln_f_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); # ln_1_g
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); # ln_1_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); # ln_2_g
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd)); # ln_2_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 3*n_embd*n_embd)); # c_attn_attn_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 3*n_embd));        # c_attn_attn_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd*n_embd));   # c_attn_proj_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, n_embd));          # c_attn_proj_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd*n_embd)); # c_mlp_fc_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        # c_mlp_fc_b

    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd*n_embd)); # c_mlp_proj_w
    ctx_size += n_layer*(ggml_row_size(GGML_TYPE_F32, 4*n_embd));        # c_mlp_proj_b

    ctx_size += n_ctx*n_layer*ggml_row_size(GGML_TYPE_F32, n_embd); # memory_k
    ctx_size += n_ctx*n_layer*ggml_row_size(GGML_TYPE_F32, n_embd); # memory_v

    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # final layer norm weight
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # final layer norm bias

    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # language model head layer norm weight
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); # language model head layer norm bias

    ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd * n_mel_vocab); # language model head linear weight
    ctx_size += ggml_row_size(GGML_TYPE_F32, n_mel_vocab); # language model head linear bias

    ctx_size += (6 + 12*n_layer)*512; # object overhead

    return ctx_size
    

class GPT2Layer:
    def __init__(self, ctx, n_embd):
        self.n_embd = n_embd

        self.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)
        self.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)

        self.ln_2_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)
        self.ln_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)

        self.c_attn_attn_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, 3*self.n_embd)
        self.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*self.n_embd)

        self.c_attn_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, self.n_embd)
        self.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)

        self.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, 4*self.n_embd)
        self.c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*self.n_embd)

        self.c_mlp_proj_w  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4*self.n_embd, self.n_embd)
        self.c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)

        

class GPT2Model:
    def __init__(self, ctx, n_ctx, n_embd, n_head, n_layer, n_text_vocab, n_mel_vocab, n_text_position, n_mel_position):
        self.tensors = {}
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_text_vocab = n_text_vocab
        self.n_text_position = n_text_position
        self.n_mel_vocab = n_mel_vocab
        self.n_mel_position = n_mel_position
        self.n_mem = n_layer * n_ctx
        self.n_elements = n_embd * n_layer * n_ctx

        self.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)
        self.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)

        self.text_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, self.n_text_vocab)
        self.text_position_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, self.n_text_position)
        self.mel_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, self.n_mel_vocab)
        self.mel_position_embedding_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, self.n_mel_position)
        self.final_layer_norm_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)
        self.final_layer_norm_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_embd)
        self.language_model_head_linear_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, self.n_embd, n_mel_vocab)
        self.language_model_head_linear_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_mel_vocab)

        self.tensors["model/ln_f/g"] = self.ln_f_g
        self.tensors["model/ln_f/b"] = self.ln_f_b

        self.tensors["model/text_embedding_weights"] = self.text_embedding_weights
        self.tensors["model/mel_embedding_weights"] = self.mel_embedding_weights
        self.tensors["model/mel_position_embedding_weights"] = self.mel_position_embedding_weights
        self.tensors["model/text_position_embedding_weights"] = self.text_position_embedding_weights
        self.tensors["model/final_layer_norm_weights"] = self.final_layer_norm_weights
        self.tensors["model/final_layer_norm_bias"] = self.final_layer_norm_bias
        self.tensors["model/language_model_head_linear_weights"] = self.language_model_head_linear_weights
        self.tensors["model/language_model_head_linear_bias"] = self.language_model_head_linear_bias

        self.gpt_layers = []

        for i in range(self.n_layer):
            layer = GPT2Layer(ctx, self.n_embd)
            self.tensors["model/h" + str(i) + "/ln_1/g"]        = layer.ln_1_g
            self.tensors["model/h" + str(i) + "/ln_1/b"]        = layer.ln_1_b

            self.tensors["model/h" + str(i) + "/ln_2/g"]        = layer.ln_2_g
            self.tensors["model/h" + str(i) + "/ln_2/b"]        = layer.ln_2_b

            self.tensors["model/h" + str(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w
            self.tensors["model/h" + str(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b

            self.tensors["model/h" + str(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w
            self.tensors["model/h" + str(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b

            self.tensors["model/h" + str(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w
            self.tensors["model/h" + str(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b

            self.tensors["model/h" + str(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w
            self.tensors["model/h" + str(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b
        
        self.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_elements)
        self.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_elements)
    
    @staticmethod
    def init_from_file(model_file: str, verbose=True, n_threads=1):
        with open(model_file, "rb") as fin:
            # Magic Number
            (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))

            assert magic == ggml.GGML_FILE_MAGIC
            if verbose:
                print("magic number =", hex(magic))
            n_mel_vocab, n_text_vocab, n_ctx, n_embd, n_head, n_layer = struct.unpack("iiiiii", fin.read(struct.calcsize("iiiiii")))

            ctx_size = compute_ctx_size(n_embd, n_layer, n_ctx, n_mel_vocab, n_text_vocab, 608, 404)
            mem_buffer = np.empty(ctx_size, dtype=np.uint8)
            init_params = ggml_init_params(
                mem_size=ctx_size,
                mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p),
            )
            ctx = ggml_init(init_params)

            model = GPT2Model(
                ctx=ctx,
                n_ctx=n_ctx,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                n_text_vocab=n_text_vocab,
                n_mel_vocab=n_mel_vocab,
                n_text_position=404,
                n_mel_position=608
            )

if __name__ == "__main__":
    model = GPT2Model.init_from_file("/home/anhnct/project/Compare_state_dict/XTTS_v2.0_original_model_files/ggml-model-f32.bin", n_threads=1)