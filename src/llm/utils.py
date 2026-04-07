import jax
import jax.numpy as jnp
from typing import NamedTuple

import tqdm
import time

import numpy as np

from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.multihost_utils import process_allgather

def as_named(x, mesh, spec):
    return jax.device_put(x, NamedSharding(mesh, spec))

def fold_in_helper(key, epoch, true_thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), true_thread_idx)

def safe_decode(tokens, tokenizer):
    try:
        stop_tokens = np.flatnonzero(tokens==0)
        if stop_tokens.size > 0:
            tokens = tokens[:stop_tokens[0]]
        return tokenizer.decode(tokens)
    except BaseException as e:
        return ""

def build_generate_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0, best_of_k=1):

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        print("compiling forward and sample")
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, generated_outs[-1] / temperature)
        else:
            sampled_tok = jnp.argmax(generated_outs[-1])
        return sampled_tok, generated_state, gen_key
    
    def generate_thread(noiser_params, params, prompt, thread_idx, epoch_num):
        print("Compiling generate_batch")

        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)
        def inner_scan(carry, input_token):
            tok, state, gen_key = carry
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo)
            return (tok, state, gen_key), true_input

        init_token = jax.lax.pvary(0, 'data')
        init_state = jax.lax.pvary(MODEL.default_state(params, config), 'data')

        def run_single_traj(gen_key):
            _, out_tokens = jax.lax.scan(inner_scan, (init_token, init_state, gen_key), prompt)
            return out_tokens

        if best_of_k == 1:
            return run_single_traj(start_gen_key)

        keys = jax.random.split(start_gen_key, best_of_k)
        out_tokens = jax.vmap(run_single_traj)(keys)
        return out_tokens

    return generate_thread


def build_generate_sft_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0):

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo, target_token):
        print("compiling forward and sample")
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        logits = generated_outs[-1]
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, logits / temperature)
        else:
            sampled_tok = jnp.argmax(logits)
        log_probs = jax.nn.log_softmax(logits)
        sampled_log_prob = log_probs[target_token]
        sampled_log_prob = jnp.where(jnp.logical_and(target_token == 0, input_token == 0), 0.0, sampled_log_prob)
        return sampled_tok, generated_state, gen_key, sampled_log_prob
    
    def generate_thread(noiser_params, params, inputs, targets, thread_idx, epoch_num, init_token, init_state):
        print("Compiling generate_batch")

        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)
        def inner_scan(carry, input_tuple):
            tok, state, gen_key = carry
            input_token, target_token = input_tuple
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key, log_prob = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo, target_token)
            return (target_token, state, gen_key), {'true_inputs': true_input, 'log_probs': log_prob}

        (last_token, last_state, _), out_dict = jax.lax.scan(inner_scan, (init_token, init_state, start_gen_key), (inputs, targets))
        out_tokens = out_dict['true_inputs']
        log_probs = out_dict['log_probs']
        return out_tokens, log_probs, last_state, last_token

    return generate_thread