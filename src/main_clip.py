# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

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
import importlib
import multiprocessing.pool
import os

from absl import app
from absl import flags
from absl import logging
from clu import parameter_overview
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from ml_collections import config_flags
import jax
import optax
import tensorflow as tf
from tensorflow.io import gfile


from src.datasets import input_pipeline
from src.evaluators import common as eval_common
from src.helpers.sharding import *
from src.helpers.utils import *
from src.losses import *
from src.optim import steps
import src.optim as optim
import src.losses as losses


tf.compat.v1.enable_eager_execution()
# prepare config


try:
    import wandb
    has_wandb = True

except ImportError:
    has_wandb = False
    print('please install wandb')

# pylint: disable=logging-fstring-interpolation


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()

jax.config.update("jax_threefry_partitionable", True)



def main(argv):
    del argv
    jax.distributed.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS",
                                                    "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

    tf.config.experimental.set_visible_devices([], "GPU")

    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir
    logging.info(  # pylint: disable=logging-fstring-interpolation
        f"\u001b[33mHello from process {jax.process_index()} holding "
        f"{jax.local_device_count()}/{jax.device_count()} devices and "
        f"writing to workdir {workdir}.\u001b[0m")

    save_ckpt_path = None
    if workdir:  # Always create if requested, even if we may not write into it.
        gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

    # The pool is used to perform misc operations such as logging in async way.
    pool = multiprocessing.pool.ThreadPool()

    # Here we register preprocessing ops from modules listed on `pp_modules`.
    for m in config.get(
            "pp_modules", [
            "ops_general", "ops_image", "ops_text"]):
        importlib.import_module(f"src.transforms.{m}")



    # This seed makes the Jax part of things (like model init) deterministic.
    # However, full training still won't be deterministic, for example due to the
    # tf.data pipeline not being deterministic even if we would set TF seed.
    # See (internal link) for a fun read on what it takes.
    rng = jax.random.PRNGKey(jax.device_put(config.get("seed", 0),
                                            jax.local_devices(backend="cpu")[0]))
    #
    # rng = jax.random.PRNGKey(config.get("seed", 0))
    ################################################################################
    #                                                                              #
    #                          Init Logging                                        #
    #                                                                              #
    ################################################################################
    xid, wid = -1, -1
    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing logging tool...")
    if config.wandb.log_wandb:
        if has_wandb and jax.process_index() == 0:
            if config.wandb.wandb_offline:
                os.environ["WANDB_MODE"] = 'offline'
            else:
                wandb.init(
                    project=str(
                        config.wandb.project), name=str(
                        config.wandb.experiment), entity=str(
                        config.wandb.entity), resume=config.wandb.resume)
                wandb.config.update(dict(config))
        else:
            logging.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # First thing after above sanity checks, so we can log "start" ticks.
    metric = BigVisionMetricWriter(xid, wid, workdir, config)


    ################################################################################
    #                                                                              #
    #                        Create Device Mesh                                    #
    #                                                                              #
    ################################################################################
    write_note("Creating mesh...")
    device_arrays = create_mesh(config)
    flatten_device_arrays = device_arrays.flatten() # this order of device can be used for pmap/loss caculation
    mesh = Mesh(device_arrays, config.sharding.mesh_axes)
    data_sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), P(*config.sharding.data_sharding))
    repl_sharding = jax.sharding.NamedSharding(mesh, P())

    repl_mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=('repl',))
    ################################################################################
    #                                                                              #
    #                        Initializing Dataloader                               #
    #                                                                              #
    ################################################################################
    write_note("Initializing train dataset...")
    batch_size = config.input.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must "
            f"be divisible by device number ({jax.device_count()})")
    info(
        "Global batch size %d on %d hosts results in %d local batch size. With "
        "%d dev per host (%d dev total), that's a %d per-device batch size.",
        batch_size,
        jax.process_count(),
        batch_size // jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        batch_size // jax.device_count())

    tokenizer = None
    if config.get('openclip_tokenizer.enable', False):
        try:
            import open_clip
        except:
            ImportError('please install openclip tp use the tokenizer')
        tokenizer  = open_clip.get_tokenizer(f'hf-hub:{config.openclip_tokenizer.repo_name}')


    # create training tfds dataset
    train_ds, ntrain_img = input_pipeline.training(config.input)

    # create dataloader
    train_iter = input_pipeline.start_input_pipeline(
        train_ds,
        config=config,
        mesh=mesh,
        data_sharding=data_sharding,
        tokenizer = tokenizer,
        )


    #calculate training steps
    total_steps = steps("total", config, ntrain_img, batch_size)
    def get_steps(name, default=ValueError, cfg=config):
        return steps(name, cfg, ntrain_img, batch_size, total_steps, default)

    chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size,
                  measure=metric.measure, write_note=write_note)

    info("Running for %d steps, that means %f epochs",
         total_steps, total_steps * batch_size / ntrain_img)

    ################################################################################
    #                                                                              #
    #                    create model/optimizer/scheduler                          #
    #                                                                              #
    ################################################################################

    write_note(f"Initializing {config.model_name} model...")
    model_mod = importlib.import_module(f"src.models.{config.model_name}")
    model = model_mod.Model(**config.get("model", {}), mesh=mesh)

    def load_any(load_config):
        """
        we load checkpoint by given load_config
        """
        model_mod = importlib.import_module(f"src.models.{config.model_name}")
        load_model = model_mod.Model(**load_config.get("model", {}), mesh=mesh)

        def init(rng):
            image_size = config.init_shapes[0]
            text_size = config.init_shapes[1]
            no_image = jnp.zeros(image_size, jnp.float32)
            no_text = jnp.zeros(text_size, jnp.int32)
            params = load_model.init(rng, no_image, no_text, train=True)["params"]
            return params

        rng, rng_init = jax.random.split(jax.random.PRNGKey(jax.device_put(config.get("seed", 0),
                                            jax.local_devices(backend="cpu")[0])))

        ## use logical sharding strategy: init parameter with sharding
        with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
            params_shape = jax.eval_shape(init, rng_init)

        params_logical_annotations = nn.get_partition_spec(params_shape)

        params_mesh_shardings = nn.logical_to_mesh_sharding(params_logical_annotations, mesh,
                                                            config.sharding.logical_axis_rules)
        params_unboxed_shape = unbox_logicallypartioned(params_shape)

        tx, sched_fns = optim.make(config, params_unboxed_shape, sched_kw=dict(
            total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

        # opt state sharding
        with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
            opt_shape = jax.eval_shape(tx.init, params_shape)
        opt_logical_annotations = nn.get_partition_spec(opt_shape)
        opt_mesh_shardings = nn.logical_to_mesh_sharding(opt_logical_annotations,
                                                         mesh,
                                                         config.sharding.logical_axis_rules)
        load_train_state_sharding = {
            "params": params_mesh_shardings,
            "opt": opt_mesh_shardings
        }
        load_train_state_shape = {
            "params": params_unboxed_shape,
            "opt": unbox_logicallypartioned(opt_shape),
        }

        def update_sharding(tree_leaf1, tree_leaf2):
            tree_leaf1.sharding = tree_leaf2

        jax.tree_util.tree_map(update_sharding, load_train_state_shape, load_train_state_sharding)

        ckpt_path = config.get('load_transform')
        abstract_train_state = jax.tree_util.tree_map(
            orbax.checkpoint.utils.to_shape_dtype_struct, load_train_state_shape
        )

        write_note(f"ft rom checkpoint {ckpt_path}...")
        ft_ckpt_mngr = create_orbax_checkpoint_manager(
            ckpt_path,
            True,
            True,
            save_interval_steps=1,
            max_to_keep=1,
        )
        latest_step = ft_ckpt_mngr.latest_step()
        write_note(f"ft rom checkpoint {ckpt_path}+{latest_step} step...")

        loaded = ft_ckpt_mngr.restore(
            latest_step,
            args=orbax.checkpoint.args.StandardRestore(abstract_train_state))
        if load_config.transform.get('patch', False):
            write_note(f'We are {load_config.transform.patch_init} patch embedding layer now')

            previous_patch_embedding = loaded['params']['img']['embedding']['kernel']
            new_kernel_size = (int(config.model.image.variant.split('/')[-1]),
                               int(config.model.image.variant.split('/')[-1]),) +previous_patch_embedding.shape[2:]

            write_note(f'Previous patch embedding layer has a shape of {previous_patch_embedding.shape}, '
                       f'we are changing to {new_kernel_size}')
            if 'inter' in load_config.transform.patch_init:
                new_patch_embedding = jax.image.resize(previous_patch_embedding, new_kernel_size, method='bilinear')

            loaded['params']['img']['embedding']['kernel'] = new_patch_embedding

        return loaded['params']



    # here we only use init function to have abstract parameter state
    def init(rng):
        image_size = config.init_shapes[0]
        text_size = config.init_shapes[1]
        no_image = jnp.zeros(image_size, jnp.float32)
        no_text = jnp.zeros(text_size, jnp.int32)
        params = model.init(rng, no_image, no_text, train=True)["params"]
        return params

    write_note("Inferring parameter shapes...")
    rng, rng_init = jax.random.split(rng)

    ## use logical sharding strategy: init parameter with sharding
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        params_shape = jax.eval_shape(init, rng_init)

    params_logical_annotations = nn.get_partition_spec(params_shape)

    with mesh, nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        params_mesh_annotations = nn.logical_to_mesh(params_logical_annotations)
    params_mesh_shardings = nn.logical_to_mesh_sharding(params_logical_annotations, mesh, config.sharding.logical_axis_rules)
    params_unboxed_shape = unbox_logicallypartioned(params_shape)

    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree.leaves(params_shape))
        metric.measure("num_params", num_params)

    write_note(f"Initializing {config.optax_name} optimizer...")
    tx, sched_fns = optim.make(config, params_unboxed_shape, sched_kw=dict(
        total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img))

    # opt state sharding
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        opt_shape = jax.eval_shape(tx.init, params_shape)
    opt_logical_annotations = nn.get_partition_spec(opt_shape)
    with mesh, nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        opt_mesh_annotations = nn.logical_to_mesh(opt_logical_annotations)
    opt_mesh_shardings = nn.logical_to_mesh_sharding(opt_logical_annotations,
                                                        mesh,
                                                        config.sharding.logical_axis_rules)

    # We jit this, such that the arrays are created on the CPU, not device[0].
    sched_fns_cpu = [jax.jit(sched_fn, backend="cpu")
                     for sched_fn in sched_fns]
    train_state_sharding = {
        "params": jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), params_mesh_annotations),
        "opt":  jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), opt_mesh_annotations)
    }

    write_note("Transferring train_state to devices...")
    # RNG is always replicated
    rng_init = reshard(rng_init, repl_sharding)

    # Parameters and the optimizer are now global (distributed) jax arrays.

    params = jax.jit(init, in_shardings=None, out_shardings=params_mesh_shardings)(rng_init)
    opt = jax.jit(tx.init, out_shardings=opt_mesh_shardings)(params)

    # this unboxing is important, otherwise the optimizer will have some issue for our customized weight decay (mask option)
    params = unbox_logicallypartioned(params)
    opt = unbox_logicallypartioned(opt)

    rng, rng_loop = jax.random.split(rng, 2)
    rng_loop = reshard(rng_loop, repl_sharding)
    del rng  # not used anymore, so delete it.

    # At this point we have everything we need to form a train state. It contains
    # all the parameters that are passed and updated by the main training step.
    train_state = {"params": params, "opt": opt}
    del params, opt  # Delete to avoid memory leak or accidental reuse.
    write_note("Logging parameter overview...")
    parameter_overview.log_parameter_overview(
        train_state["params"], msg="Init params",
        include_stats="global", jax_logging_process=0)
    ################################################################################
    #                                                                              #
    #                                 Update Step                                  #
    #                                                                              #
    ################################################################################
    @functools.partial(
        jax.jit,
        donate_argnums=(0,),
        in_shardings=(train_state_sharding, data_sharding, repl_sharding),
        out_shardings=(train_state_sharding, repl_sharding))
    def update_fn(train_state, batch, rng):
        """Update step."""
  
        images = batch["image"]
        labels_key1 = batch["labels1"]
        labels_key2 = batch["labels2"]
        # 在 batch 维度 (axis=0) 上合并
        labels = jnp.concatenate([labels_key1, labels_key2], axis=0)

        if config.get("cpu_unit8", False):
            mean = jnp.asarray(
                [0.485 * 255, 0.456 * 255, 0.406 * 255])[None, None, None, :]
            std = jnp.asarray(
                [0.229 * 255, 0.224 * 255, 0.225 * 255])[None, None, None, :]
            images = (jnp.asarray(images, dtype=jnp.float32) - mean) / std


        # # Get device-specific loss rng for each step
        step_count = optim.get_count(train_state["opt"], jittable=True)

        rng = jax.random.fold_in(rng, step_count)

        rng, rng_model = jax.random.split(rng, 2)


        def loss_fn(params, images, labels):


            zimg, ztxt, extras = model.apply({"params": params},
                                             images, labels,
                                             train=True,
                                             rngs={
                "dropout": rng, 'drop_path': rng, 'random_mask': rng})

            if config.get("local_loss", False):
                local_img, local_txt = zimg, ztxt
            else:
                local_img, local_txt = None, None


            loss_type = config.get("loss_type", "clip")
            if loss_type == "clip":
                l, l_extras = losses.bidirectional_contrastive_loss(
                    zimg, ztxt, extras["t"], reduction=True, local_loss=config.local_loss, local_img_logits=local_img, local_txt_logits=local_txt, devices=flatten_device_arrays)
            elif loss_type == "coca":
                half_batch_size = ztxt.shape[0] // 2

                # split ztxt into two parts
                ztxt_1, ztxt_2 = ztxt[:half_batch_size], ztxt[half_batch_size:]
                l, l_extras = losses.bidirectional_contrastive_loss(
                    zimg, ztxt_1, ztxt_2, extras["t"], reduction=True, local_loss=config.local_loss, local_img_logits=local_img, local_txt_logits=local_txt, devices=flatten_device_arrays)

                # autoregression text prediction loss
                autoreg_labels = batch["autoreg_labels"]
                logits_txt = extras[f"logits"]
                cap_loss_mask = batch["cap_loss_mask"]
                caption_l = losses.softmax_xent(logits=logits_txt, labels=autoreg_labels, mask=cap_loss_mask, reduction=True, axis=-1)
                clip_loss_weight = config.get("clip_loss_weight")
                cap_loss_weight = config.get("coca_caption_loss_weight")
                l_extras["clip_loss"] = l
                l_extras["caption_loss"] = caption_l
                l = clip_loss_weight * l + cap_loss_weight * caption_l
            else:
                raise ValueError


            return l, {
                "t": extras["t"],
                "t/parameter": extras["t/parameter"],
                "nimg": jnp.mean(extras["img/norm"]),
                "ntxt": jnp.mean(extras["txt/norm"]),
                **l_extras,
            }

        params, opt = train_state["params"], train_state["opt"]

        (l, measurements), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, images, labels)
        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)

        measurements["training_loss"] = l
        gs = jax.tree.leaves(optim.replace_frozen(config.schedule, grads, 0.))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree.leaves(params)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        us = jax.tree.leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

        return{"params": params, "opt": opt}, measurements


    ################################################################################
    #                                                                              #
    #                               Load Checkpoint                                #
    #                                                                              #
    ################################################################################

    # Decide how to initialize training. The order is important.
    # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
    # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
    # 3. Initialize model from something, e,g, start a fine-tuning job.
    # 4. Train from scratch.
    resume_ckpt_path = None
    if save_ckpt_path and gfile.exists(f"{save_ckpt_path}"):
        resume_ckpt_path = save_ckpt_path
    elif config.get("resume"):
        resume_ckpt_path = fillin(config.resume)

    ckpt_mngr = None
    if save_ckpt_path or resume_ckpt_path:
        #ckpt_mngr = array_serial.GlobalAsyncCheckpointManager()
        # use obrax checkpoint
        ckpt_mngr = create_orbax_checkpoint_manager(
                save_ckpt_path,
                True,
                True,
                save_interval_steps=1,
                max_to_keep=1,
            )

    if ckpt_mngr:
        latest_step = ckpt_mngr.latest_step()
        if latest_step:
            # resume
            write_note(f"Resuming training from checkpoint {resume_ckpt_path}...")
            abstract_train_state = jax.tree_util.tree_map(
                orbax.checkpoint.utils.to_shape_dtype_struct, train_state
            )

            train_state = ckpt_mngr.restore(
                latest_step,
                args= orbax.checkpoint.args.StandardRestore(abstract_train_state),
                )


            chrono_ckpt_path = save_ckpt_path.replace('checkpoint.npz', 'chrono.npz')
            chrono_checkpoint = {
                "chrono": chrono.save(),
            }
            chrono_checkpoint = jax.tree_structure(chrono_checkpoint)
            chrono_loaded = load_checkpoint(chrono_checkpoint, chrono_ckpt_path)
            chrono.load(chrono_loaded["chrono"])


        elif config.get('ft_from', None):
            ckpt_path = config.get('ft_from')
            abstract_train_state = jax.tree_util.tree_map(
                orbax.checkpoint.utils.to_shape_dtype_struct, train_state
            )

            write_note(f"ft rom checkpoint {ckpt_path}...")
            ft_ckpt_mngr = create_orbax_checkpoint_manager(
                    ckpt_path,
                    True,
                    True,
                    save_interval_steps=1,
                    max_to_keep=1,
                )
            latest_step = ft_ckpt_mngr.latest_step()
            write_note(f"ft rom checkpoint {ckpt_path}+{latest_step} step...")

            loaded = ft_ckpt_mngr.restore(
                latest_step,
                args= orbax.checkpoint.args.StandardRestore(abstract_train_state))
            #TODO we can do some interpolation with the weights here

            # load weight manually
            train_state['params'] = loaded['params']
            del loaded

            # after loading we log the parameter for sanity check
            write_note("Logging parameter overview...")
            parameter_overview.log_parameter_overview(
                train_state["params"], msg="Init params",
                include_stats="global", jax_logging_process=0)
        elif config.get('load_transform', None):
            loaed_transformed_ckpt = load_any(config.load_config)
            train_state['params'] = loaed_transformed_ckpt
            del loaed_transformed_ckpt

            # after loading we log the parameter for sanity check
            write_note("Logging parameter overview...")
            parameter_overview.log_parameter_overview(
                train_state["params"], msg="Init params",
                include_stats="global", jax_logging_process=0)


        elif config.get("masked_init"):
            write_note(f"Initialize model from {config.masked_init}...")
            pretrained_params_cpu = load_params(None, config.masked_init)

            params_cpu = jax.tree.map(recover_dtype, pretrained_params_cpu)
            # TODO: when updating the `load` API soon, do pass and request the


            # load has the freedom to return params not correctly sharded. Think of for
            # example ViT resampling position embedings on CPU as numpy arrays.
            train_state["params"] = reshard(
                params_cpu, params_mesh_shardings)

            # parameter_overview.log_parameter_overview(
            #     train_state["params"], msg="restored params",
            #     include_stats="global", jax_logging_process=0)


    ################################################################################
    #                                                                              #
    #                                 Setup Evals                                  #
    #                                                                              #
    ################################################################################
    # We require hashable function reference for evaluator.
    # We do not jit/pmap this function, because it is passed to evaluator that
    # does it later. We output as many intermediate tensors as possible for
    # maximal flexibility. Later `jit` will prune out things that are not
    # needed.
    def eval_logits_fn(train_state, batch):
        zimg, ztxt, out = model.apply(
            {"params": train_state["params"]},
            batch.get("image", None), batch.get("labels", None))
        return zimg, ztxt, out

    eval_fns = {
        "predict": eval_logits_fn,
    }
    # Only initialize evaluators when they are first needed.
    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config, eval_fns,
            lambda s: write_note(f"Init evaluator: {s}…\n{chrono.note}"),
            lambda key, cfg: get_steps(key, default=None, cfg=cfg),
            mesh,
            data_sharding,
            train_state_sharding['params'],
            tokenizer=tokenizer
        )

    ################################################################################
    #                                                                              #
    #                                 Start Training                               #
    #                                                                              #
    ################################################################################
    write_note("Kicking off misc stuff...")
    first_step_device = optim.get_count(train_state["opt"], jittable=True)
    first_step = int(jax.device_get(first_step_device))
    chrono.inform(first_step=first_step)
    prof = None  # Keeps track of start/stop of profiler state.


    if config.get('eval_only', False):
        step = 0
        for (name, evaluator, _, prefix) in evaluators():
            chrono.pause(wait_for=train_state)
            # Record things like epoch number, core hours etc.
            chrono.tick(step)
            write_note(f"{name} evaluation...\n{chrono.note}")
            with chrono.log_timing(f"z/secs/eval/{name}"):
                with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
                    for key, value in evaluator.run(train_state):
                        metric.measure(f"{prefix}{key}", value)
            chrono.resume()
        metric.step_end()
        exit()


    write_note(f"First step compilations...\n{chrono.note}")
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        metric.step_start(step)
        jax.experimental.multihost_utils.sync_global_devices('data_loading')

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):

                with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
                    train_state, measurements = update_fn(train_state, batch, rng_loop)

        # On the first host, let's always profile a handful of early steps.
        if jax.process_index() == 0:
            prof = startstop_prof(
                prof, step, first_step, get_steps("log_training"))

        # Report training progress
        if (itstime(step, get_steps("log_training"), total_steps, host=0)
                or chrono.warmup and jax.process_index() == 0):
            for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                metric.measure(
                    f"global_schedule{i if i else ''}",
                    sched_fn_cpu(
                        jax.device_put(step - 1,
                                       jax.local_devices(backend="cpu")[0])))
            measurements = jax.device_get(measurements)

            for name, value in measurements.items():
                metric.measure(name, value)
            chrono.tick(step)

        jax.experimental.multihost_utils.sync_global_devices('reporting')

        # save ckpt
        keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
        if save_ckpt_path and (
                (keep := itstime(step, keep_ckpt_steps, total_steps, first=False))
                or itstime(step, get_steps("ckpt", None), total_steps, first=True)
        ):

            chrono.pause(wait_for=(train_state))


            ckpt = {**train_state}

            ckpt_mngr.save(
                step, args=orbax.checkpoint.args.StandardSave(ckpt))


            jax.experimental.multihost_utils.sync_global_devices('final_eval')

            chrono_ckpt_path =  save_ckpt_path.replace('checkpoint.npz', 'chrono.npz')
            chronockpt = {
                "chrono": chrono.save()}
            ckpt_writer = pool.apply_async(
                save_checkpoint, (chronockpt, chrono_ckpt_path, None))
            chrono.resume()
            ckpt_mngr.wait_until_finished()


        for (name, evaluator, log_steps, prefix) in evaluators():
            if itstime(
                    step,
                    log_steps,
                    total_steps,
                    first=False,
                    last=True):
                chrono.pause(wait_for=train_state)
                # Record things like epoch number, core hours etc.
                chrono.tick(step)
                write_note(f"{name} evaluation...\n{chrono.note}")
                with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
                    for key, value in evaluator.run(train_state):
                        metric.measure(f"{prefix}{key}", value)
                chrono.resume()
        metric.step_end()
        jax.experimental.multihost_utils.sync_global_devices('eval')

        if has_wandb and jax.process_index() == 0:
            if config.wandb.log_wandb:
                wandb.log(metric.step_metrics, step=step)
        jax.experimental.multihost_utils.sync_global_devices('wandb_log')

    # Run evals after done with training. Running them here guarantees evals
    # will run if job is restarted after writting the last checkpoint and
    # also supports eval only runs (when total_steps or num_epochs is 0).
    metric.step_start(total_steps)
    for (name, evaluator, _, prefix) in evaluators():
        with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
            for key, value in evaluator.run(train_state):
                metric.measure(f"{prefix}{key}", value)
    if has_wandb and jax.process_index() == 0:
        if config.wandb.log_wandb:
            wandb.log(metric.step_metrics, step=total_steps)
    # Always give a chance to stop the profiler, no matter how things ended.
    # TODO: can we also do this when dying of an exception like OOM?
    if jax.process_index() == 0 and prof is not None:
        startstop_prof(prof)
    jax.experimental.multihost_utils.sync_global_devices('final_eval')

    # Last note needs to happen before the pool's closed =)
    write_note(f"Done!\n{chrono.note}")

    pool.close()
    pool.join()
    metric.close()

    maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)

    jax.experimental.multihost_utils.sync_global_devices('done')


if __name__ == "__main__":
    app.run(main)
