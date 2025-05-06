# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.
#
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

import functools
import logging
import sys
import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
import flax.jax_utils as flax_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

from src.helpers.sharding import reshard
from src.helpers.utils import onehot




def unbatched_gather(x, ids_keep):
    return x[ids_keep, Ellipsis]


batched_gather = jax.vmap(unbatched_gather)

def sigmoid_xent(*, logits, labels, reduction=True):
  # NOTE: This implementation is stable, see these two:
  # (internal link)
  # https://github.com/google/jax/issues/2140
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  nll = -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)
  return jnp.mean(nll) if reduction else nll


def bidirectional_contrastive_loss(
        zimg,
        ztxt_1,
        ztxt_2,
        t,
        devices=None,
        mask=None,
        reduction=False,
        local_loss=True,
        global_loss=False,
        efficient_loss=False,
        local_img_logits=None,
        local_txt_logits=None,):
    """Bidirectional contrastive losses (e.g. for contrastive trainer/evaluator)."""
    # BF.FB = BB
    if global_loss:
        logits = jnp.dot(zimg, ztxt.T) * t


        if mask is not None:
           
            exclude = jnp.logical_not(mask)  # Now 1 if we don't want to keep.
            exclude = jnp.logical_or(exclude[:, None], exclude[None, :])
            logits = jnp.where(exclude, jnp.NINF, logits)

        # Note: assumed t is in a good range e.g. already passed through
        # exp/softplus.
        l1 = -jnp.diag(jax.nn.log_softmax(logits, axis=1))  # NLL img->txt
        l2 = -jnp.diag(jax.nn.log_softmax(logits, axis=0))  # NLL txt->img
        l = 0.5 * (l1 + l2)

        if mask is not None:
            l = jnp.where(mask, l, 0)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

    elif efficient_loss:
        # This implementation is based on FLIP https://github.com/facebookresearch/flip
        # memory-efficient implementation
        logits = jnp.einsum("nc,mc->nm", zimg, ztxt)
        logging.info("logits.shape: {}".format(logits.shape))
        logits *= t

        # ---------------------------------------------------------------------------
        logits_pos = jnp.einsum(
            "nc,nc->n", zimg, ztxt
        )  # easier to take the diagonal (positive)
        logits_pos *= t

        # hand-written log_softmax
        # we do not need to shift x_max as it is well-bound after l2-normalization
        exp_logits = jnp.exp(logits)
        logsumexp_logits01 = jnp.log(jnp.sum(exp_logits, axis=-1))  # [N,]
        logsumexp_logits10 = jnp.log(jnp.sum(exp_logits, axis=0))  # [N,]

        loss01 = -(logits_pos - logsumexp_logits01)  # [N,]
        loss10 = -(logits_pos - logsumexp_logits10)  # [N,]

        loss01 = loss01.mean()
        loss10 = loss10.mean()

        l = (loss01 + loss10) / 2
        if mask is not None:
            l = jnp.where(mask, l, 0)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)
    elif local_loss:

    
        mesh = Mesh(devices, axis_names=('batch',))
        @partial(shard_map, mesh=mesh,  in_specs=(P('batch', None), P('batch', None), P('batch', None)),
                 out_specs=(P('batch', None)))
        def local_logits(local_img, local_txt1, local_txt2):
            # perform all_gather operation on local_txt1 and local_txt2
            ztxt1 = jax.lax.all_gather(local_txt1, 'batch')
            ztxt2 = jax.lax.all_gather(local_txt2, 'batch')
            zimg = jax.lax.all_gather(local_img, 'batch')

            # adjust the shape
            ztxt1 = ztxt1.reshape((jax.device_count() * local_txt1.shape[0], local_txt1.shape[1]))
            ztxt2 = ztxt2.reshape((jax.device_count() * local_txt2.shape[0], local_txt2.shape[1]))
            zimg = zimg.reshape((jax.device_count() * local_img.shape[0], local_img.shape[1]))

            print('gather shape is:', zimg.shape)

            # calculate the logits of local_img and ztxt1
            logits_img1 = jax.nn.log_softmax(
                jnp.dot(local_img, ztxt1.T) * t, axis=1)
            logits_txt1 = jax.nn.log_softmax(
                jnp.dot(local_txt1, zimg.T) * t, axis=1)

            # calculate the logits of local_img and ztxt2
            logits_img2 = jax.nn.log_softmax(
                jnp.dot(local_img, ztxt2.T) * t, axis=1)
            logits_txt2 = jax.nn.log_softmax(
                jnp.dot(local_txt2, zimg.T) * t, axis=1)

            rank = jax.lax.axis_index('batch')
            print(f'local logits has a shape of {logits_img1.shape}')

            # calculate the loss of the first part
            l1_part1 = -jnp.array([logits_img1[i][i + rank * logits_img1.shape[0]]
                                    for i in range(logits_img1.shape[0])])
            l2_part1 = -jnp.array([logits_txt1[i][i + rank * logits_txt1.shape[0]]
                                    for i in range(logits_txt1.shape[0])])

            # calculate the loss of the second part
            l1_part2 = -jnp.array([logits_img2[i][i + rank * logits_img2.shape[0]]
                                    for i in range(logits_img2.shape[0])])
            l2_part2 = -jnp.array([logits_txt2[i][i + rank * logits_txt2.shape[0]]
                                    for i in range(logits_txt2.shape[0])])

            # combine the loss of the two parts
            local_loss1 = 0.5 * (l1_part1 + l2_part1)
            local_loss2 = 0.5 * (l1_part2 + l2_part2)

            # calculate the average loss
            local_loss = jnp.mean((local_loss1 + local_loss2) / 2, keepdims=True)
            # local_loss = jnp.mean((0.6 * local_loss1 + 0.4 * local_loss2), keepdims=True)
            local_loss = jax.lax.pmean(local_loss, 'batch')

            print(f'local loss has a shape of {local_loss.shape}')
            return local_loss

        l = local_logits(zimg, ztxt_1, ztxt_2)



        if mask is not None:
            l = jnp.where(mask, l, 0)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x):
                return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)
        return redux(l), {"ncorrect": 0 }

    else:
        # deprecated
        rank = jax.lax.axis_index('batch')
        logits_img = jax.nn.log_softmax(
            jnp.dot(local_img_logits, ztxt.T) * t, axis=1)
        logits_txt = jax.nn.log_softmax(
            jnp.dot(local_txt_logits, zimg.T) * t, axis=1)

        l1 = -jnp.array([logits_img[i][i + rank * logits_img.shape[0]]
                         for i in range(logits_img.shape[0])])
        l2 = -jnp.array([logits_txt[i][i + rank * logits_txt.shape[0]]
                         for i in range(logits_txt.shape[0])])

        l = 0.5 * (l1 + l2)

        redux = jnp.mean if reduction else lambda x: x
        if reduction and mask is not None:
            def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

        return redux(l), {
            "ncorrect": redux(
                jnp.argmax(
                    logits_img, axis=1) == jnp.arange(
                    len(logits_img))), }

    # Also return extra measurements.
    return redux(l), {
        "ncorrect": redux(
            jnp.argmax(
                logits, axis=1) == jnp.arange(
                len(logits))), }



def softmax_xent(*, logits, labels, mask=None, reduction=True, kl=False, axis=-1, smoothing=0.1):

    # one-hot encode the labels
    vocab_size = logits.shape[axis]
    # labels = labels[:, :-1] 

    # labels = jnp.pad(labels, ((0, 0), (0, 1)), constant_values=0)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    


    # calculate the log-softmax
    log_p = jax.nn.log_softmax(logits, axis=axis)

    # calculate the cross-entropy loss
    nll = -jnp.sum(one_hot_labels * log_p, axis=axis)

    # if using KL divergence, also calculate the entropy of the labels
    if kl:
        nll += jnp.sum(one_hot_labels * jnp.log(jnp.clip(one_hot_labels, 1e-8)), axis=axis)

    # use the custom reduction mechanism
    redux = jnp.mean if reduction else lambda x: x
    if reduction and mask is not None:
        def redux(x): return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

    return redux(nll)




def bce_logits(*, logits, labels, weight=None, reduction=True):

    def bce(logits, labels, weight=None, reduction=True):
        """
        Binary Cross Entropy Loss
        Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
        :param x: Input tensor
        :param y: Target tensor
        :param weight: Vector of example weights
        :param average: Boolean to average resulting loss vector
        :return: Scalar value
        """
        max_val = jnp.clip(logits, 0, None)
        loss = logits - logits * labels + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-logits - max_val)))

        if weight is not None:
            loss = loss * weight

        if reduction:
            return loss.mean()
        else:
            return loss
    return  jnp.mean(jax.vmap(bce)(logits, labels))


def weighted_softmax_xent(*,
                          logits,
                          labels,
                          reduction=True,
                          weights=None,
                          label_smoothing=0.0,
                          normalize=True):
  """Compute weighted cross entropy.

  Args:
   logits: [batch, length, num_classes] float array.
   labels: categorical targets [batch, length] int array.
   reduction: reduce across batch dim.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   normalize: normalize each "sentence" losses by the number of tokens in it.

  Returns:
    Tuple of scalar losses and batch normalizing factor.
  """
  if logits.ndim != labels.ndim + 1:
    raise ValueError("Incorrect shapes. Got shape %s logits and %s targets" %
                     (str(logits.shape), str(labels.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  soft_targets = onehot(
      labels, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)

  normalizing_factor = labels.shape[1]
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum(axis=1)

  loss = loss.sum(axis=1)
  if normalize:
    loss = loss / normalizing_factor

  return loss.mean() if reduction else loss




def mae_loss(*, pred, target, mask, norm_pix_loss: bool = True):
    if norm_pix_loss:
        mean = target.mean(axis=-1, keepdims=True)
       # var = target.var(axis=-1, keepdims=True)
        var = target.var(axis=-1, keepdims=True)  * target.shape[-1]/(target.shape[-1]-1) # unbiased version
        target = (target - mean) / (var + 1e-6) ** .5
    loss = (pred - target) ** 2

    loss = loss.mean(axis=-1)

    loss = (loss * mask).sum() / mask.sum()


    return loss