# Convert GPT-2 huggingface transformer model to ggml format
#
# Load the model using GPT2Model.
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import struct
import json
import numpy as np
import re
import torch

from transformers import GPT2Model

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 2:
    print("Usage: convert-h5-to-ggml.py dir-model [use-f32]\n")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]
fname_out = sys.argv[1] + "/ggml-model.bin"

with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

# with open(dir_model + "/added_tokens.json", "r", encoding="utf-8") as f:
#     encoder_added = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 2:
    use_f16 = False
    fname_out = sys.argv[1] + "/ggml-model-f32.bin"

#model = GPT2Model.from_pretrained(dir_model)
model = torch.load(dir_model + "/model.pth")

list_vars = model["model"]
#print (list_vars)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["model_args"]["gpt_num_audio_tokens"]))
fout.write(struct.pack("i", hparams["model_args"]["gpt_number_text_tokens"]))
fout.write(struct.pack("i", hparams["model_args"]["gpt_max_text_tokens"] + hparams["model_args"]["gpt_max_audio_tokens"] + 32))
fout.write(struct.pack("i", hparams["model_args"]["gpt_n_model_channels"]))
fout.write(struct.pack("i", hparams["model_args"]["gpt_n_heads"]))
fout.write(struct.pack("i", hparams["model_args"]["gpt_layers"]))
#fout.write(struct.pack("i", hparams["rotary_dim"]))
#fout.write(struct.pack("i", use_f16))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

#fout.write(struct.pack("i", len(encoder)))

# for key in encoder:
#     text = bytearray([byte_decoder[c] for c in key])
#     fout.write(struct.pack("i", len(text)))
#     fout.write(text)

# for key in encoder_added:
#     text = bytearray([byte_decoder[c] for c in key])
#     fout.write(struct.pack("i", len(text)))
#     fout.write(text)

for name in list_vars.keys():
    if "gpt" in name and "conditioning" not in name and "text_head" not in name:
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: " + name + " with shape: ", data.shape)

        # we don't need these
        if name.endswith("attn.masked_bias") or name.endswith(".attn.bias"):
            print("  Skipping variable: " + name)
            continue

        n_dims = len(data.shape);

        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype = 0;
        if use_f16:
            if name != "wpe.weight" and (name[-7:] == ".weight" and n_dims == 2):
                print("  Converting to float16")
                data = data.astype(np.float16)
                ftype = 1
            else:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0

        # for efficiency - transpose these matrices:
        #  "transformer.h.*.mlp.c_proj.weight
        if name.endswith(".mlp.c_proj.weight") or name.endswith("c_attn.weight") or name.endswith("c_fc.weight") or name.endswith("c_proj.weight"):
            print("  Transposing")
            data = data.transpose()

        # rename headers to keep compatibility
        name = name.replace("gpt.gpt.", "")
        name = name.replace("gpt.", "")

        if name == "ln_f.weight":
            name = "model/ln_f/g"
        elif name == "ln_f.bias":
            name = "model/ln_f/b"
        elif re.match(r"h\.\d+\.ln_1\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/g"
        elif re.match(r"h\.\d+\.ln_1\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_1/b"
        elif re.match(r"h\.\d+\.attn\.c_attn\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/w"
        elif re.match(r"h\.\d+\.attn\.c_attn\.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_attn/b"
        elif re.match(r"h\.\d+\.attn\.c_proj\.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/w"
        elif re.match(r"h.\d+.attn.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/attn/c_proj/b"
        elif re.match(r"h.\d+.ln_2.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/g"
        elif re.match(r"h.\d+.ln_2.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/ln_2/b"
        elif re.match(r"h.\d+.mlp.c_fc.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/w"
        elif re.match(r"h.\d+.mlp.c_fc.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_fc/b"
        elif re.match(r"h.\d+.mlp.c_proj.weight", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/w"
        elif re.match(r"h.\d+.mlp.c_proj.bias", name):
            i = re.findall("\d+", name)[0]
            name = f"model/h{i}/mlp/c_proj/b"
        elif name == "text_embedding.weight":
            name = "model/text_embedding_weights"
            #name = "model/text_embd_w"
        elif name == "mel_embedding.weight":
            name = "model/mel_embedding_weights"
        elif name == "mel_pos_embedding.emb.weight":
            name = "model/mel_position_embedding_weights"
        elif name == "text_pos_embedding.emb.weight":
            name = "model/text_position_embedding_weights"
        elif name == "final_norm.weight":
            name = "model/final_layer_norm_weights"
        elif name == "final_norm.bias":
            name = "model/final_layer_norm_bias"
        elif name == "mel_head.weight":
            name = "model/language_model_head_linear_weights"
        elif name == "mel_head.bias":
            name = "model/language_model_head_linear_bias"
        else:
            print("Unrecognized variable name. %s", name)

        str = name.encode('utf-8')

        fout.write(struct.pack("iii", n_dims, len(str), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str);

        # data
        data.tofile(fout)

# fout.close()

# print("Done. Output file: " + fname_out)
# print("")