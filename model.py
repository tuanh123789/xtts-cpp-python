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

from ggml import ggml_new_tensor_1d, ggml_new_tensor_2d, GGML_TYPE_F32, ggml_row_size, ggml_init_params, ggml_init, ggml_nbytes, ggml_get_data, \
    ggml_new_graph, GGML_TYPE_I32, ggml_add, ggml_get_rows, ggml_norm, ggml_mul, ggml_mul_mat, ggml_repeat, ggml_view_2d, ggml_view_1d,ggml_cpy, \
    ggml_permute, ggml_new_tensor_3d, ggml_reshape_3d, ggml_view_1d, ggml_element_size, ggml_scale_inplace, ggml_soft_max_inplace

from ggml.utils import from_numpy, to_numpy

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
            self.gpt_layers.append(layer)
        
        self.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_elements)
        self.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, self.n_elements)
    
    def generate(self, n_past, input_ids):

        N = len(input_ids)
        buf_size = 512 * 1024 * 1024  # 512MB
        buf = np.empty(buf_size, dtype=np.uint8)

        params = ggml_init_params(
            mem_size=buf_size,
            mem_buffer=None,
        )

        ctx0 = ggml_init(params)

        embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N)
        position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N)
        embd = from_numpy(np.array(input_ids, dtype=np.int32), ctx0)
        position = from_numpy(np.arange(0, N, 1, dtype=np.int32), ctx0)

        inpL = ggml_add(ctx0, ggml_get_rows(ctx0, self.text_embedding_weights, embd), ggml_get_rows(ctx0, self.text_position_embedding_weights, position))
        
        for i in range(self.n_layer):
            cur = ggml_norm(ctx0, inpL, 1e-05)
            cur = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, self.gpt_layers[i].ln_1_g, cur), cur), ggml_repeat(ctx0, model.gpt_layers[i].ln_1_b, cur))

            cur = ggml_mul_mat(ctx0, self.gpt_layers[i].c_attn_attn_w, cur)
            cur = ggml_add(ctx0, ggml_repeat(ctx0, self.gpt_layers[i].c_attn_attn_b, cur), cur)

            Qcur = ggml_view_2d(ctx0, cur, self.n_embd, N, cur.tensor.contents.nb[1], 0 * ctypes.sizeof(ctypes.c_float) * self.n_embd)

        gf = ggml_new_graph(ctx0)

        ggml.ggml_build_forward_expand(gf, cur)
        ggml.ggml_graph_compute_with_ctx(ctx0, gf, 1)
        
        print(to_numpy(cur))
    
    def main(self):
        input_ids = [261, 259, 62, 84, 28, 2, 125, 2, 27, 5355, 2, 54, 2, 1108, 351, 0, 1024]
        n_predict = 200
        n_past = 0
        embd = []

        for i in range (len(embd), len(input_ids) + n_predict):
            if len(embd) > 0:
                self.generate(n_past, embd)

            n_past += len(embd)
            embd.clear()

            if i >= len(input_ids):
                pass
            else:
                for k in range(i, len(input_ids)):
                    embd.append(input_ids[k])
                i += len(input_ids) - 1


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
                mem_buffer=None,
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

            # Read n_dims, length, and ttype

            while True:
                nbytes = struct.calcsize("iii")
                data = fin.read(nbytes)
                if len(data) != nbytes:
                    break
                (n_dims, s_len, ftype) = struct.unpack("iii", data)
                dims = struct.unpack(
                    "i" * n_dims, fin.read(struct.calcsize("i" * n_dims))
                )
                tensor_name = fin.read(s_len).decode("utf-8")
                tensor = model.tensors[tensor_name]
            
                buf = (ctypes.c_char * ggml_nbytes(tensor)).from_address(ggml_get_data(tensor))
                offset = fin.tell()
                fname = fin.name.encode("utf-8")
                fin.readinto(buf)

            return model

if __name__ == "__main__":
    model = GPT2Model.init_from_file("/home/anhnct/project/Compare_state_dict/XTTS_v2.0_original_model_files/ggml-model-f32.bin", n_threads=1)
    model.generate(n_past=0, input_ids=[261, 259, 62, 84, 28, 2, 125, 2, 27, 5355, 2, 54, 2, 1108, 351, 0, 1024])