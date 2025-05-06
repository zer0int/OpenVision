# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer encoders both for text and for images."""


import importlib
import sys
import time
from typing import Any, Optional, Tuple, Union


import flax.linen as nn
import jax.numpy as jnp
import numpy as np


from src.helpers import utils

ConfigDict = Any
class Model(nn.Module):
    out_dim: Union[int, Tuple[int, int]] = 512
    text: Optional[ConfigDict] = None
    image: Optional[ConfigDict] = None
    quick_gelu: bool = False
    temperature_init: float = 10.0
    init_logit_bias: Optional[float] = None
    cast_dtype: Optional[jnp.dtype] = None
    pad_id: int = 0
    text_model: str = "proj.image_text.text_transformer"
    image_model: str = "vit"
    mesh: Any = None
    text_decoder_config: Optional[ConfigDict] = None
    text_decoder: Optional[str] = None

    @nn.compact
    def __call__(self, image, text=None, image_latent=None, image_embs=None, output_labels=True, train=False, **kw):
        ztxt, zimg = None, None
        out_dims = self.out_dim
        out_dict = {}
        out_dict["logits"] = None
        out_dict["logit_bias"] = None
        labels = None
        if isinstance(out_dims, int):
            out_dims = (out_dims, out_dims)


        if image is not None:
            image_model = importlib.import_module(f"src.models.{self.image_model}").Model(
                **{"num_classes": out_dims[0], **(self.image or {})}, name="img", mesh=self.mesh, **kw)
            if image_latent is None or image_embs is None:
                zimg, image_embs = image_model(image, train=train, **kw)
            zimg = zimg.astype(jnp.float32)
             # Normalize the embeddings the models give us.
            zimg = nn.with_logical_constraint(zimg, ("activation_batch", "activation_embed"))
            out_dict["img/norm"] = jnp.linalg.norm(zimg, axis=1, keepdims=True)
            out_dict["img/normalized"] = zimg = zimg / (out_dict["img/norm"] + 1e-8)
            zimg = nn.with_logical_constraint(zimg, ("activation_batch", "activation_embed"))

        if text is not None:
            text_model = importlib.import_module(
                f"src.models.{self.text_model}"
            ).Model(**{"num_classes": out_dims[1], **(self.text or {})}, name="txt", mesh=self.mesh)

        if text is not None:
            ztxt, token_embs = text_model(text, **kw)
            ztxt = ztxt.astype(jnp.float32)

            ztxt = nn.with_logical_constraint(ztxt, ("activation_batch", "activation_embed"))

            # Normalize the embeddings the models give us.
            out_dict["txt/norm"] = jnp.linalg.norm(ztxt, axis=1, keepdims=True)
            out_dict["txt/normalized"] = ztxt = ztxt / (out_dict["txt/norm"] + 1e-8)
            ztxt = nn.with_logical_constraint(ztxt, ("activation_batch", "activation_embed"))
      

        if (text is not None) and (image is not None) and (self.text_decoder != 'none'):
            text_decoder = importlib.import_module(f"src.models.{self.text_decoder}").Model(
                **(self.text_decoder_config or {}), name="txt_decoder", **kw)  # pylint: disable=not-a-mapping

            if train:

                token_embs = token_embs[:token_embs.shape[0] // 2]
            logits, out_decoder_txt = text_decoder(image_embs, token_embs, train=train, **kw)
            out_dict["logits"] = logits

        # Initialize logit_scale and logit_bias in __call__
        temp_init = jnp.log(self.temperature_init)
        t = self.param("t", lambda key, shape, dtype: temp_init *
                    jnp.ones(shape, dtype), (1,), jnp.float32)
        out_dict["t"] = jnp.exp(t)
        out_dict["t/parameter"] = t

 
        if (b_init := self.init_logit_bias) is not None:
            out_dict["b"] = self.param("b", lambda k, s, d: b_init * jnp.ones(s, d),
                                (1,), jnp.float32)

        if labels is not None:
            out_dict["labels"] = labels
        if b_init is not None:
            out_dict["logit_bias"] = b_init

        return zimg, ztxt, out_dict


def load(init_params, init_files, model_cfg, img_load_kw={}, txt_load_kw={}):  # pylint: disable=dangerous-default-value
    """Loads both towers, `init_files` is now a dict with `img` and `txt` keys."""
    if isinstance(init_files, str):
        # A shortcut for a single file checkpoint of a two_towers model.
        init_files = {k: f"{init_files}:{k}" for k in ("img", "txt", "t")}
    else:
        # Shallow copy because we'll pop stuff off.
        init_files = {**init_files}

    restored_params = {**init_params}

    img_init = init_files.pop("image", init_files.pop("img", None))
    if img_init:
        restored_params["img"] = importlib.import_module(
            f"src.models.{model_cfg.image_model}"
        ).load(init_params["img"], img_init, model_cfg.image, **img_load_kw)

    txt_init = init_files.pop("text", init_files.pop("txt", None))
    if txt_init:
        restored_params["txt"] = importlib.import_module(
            f"src.models.{model_cfg.text_model}"
        ).load(init_params["txt"], txt_init, model_cfg.text, **txt_load_kw)

    t_init = init_files.pop("temperature", init_files.pop("t", None))
    if t_init:
        restored_params["t"] = utils.load_params(None, t_init)

    assert not init_files, (
        f"There's something unused left in `config.model_init`. You probably got "
        f"a typo. Here it is: {init_files}")

    return restored_params
