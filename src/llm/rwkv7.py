import jax
import jax.numpy as jnp

from functools import partial

from .llm import LLM
from .base_model import Model, CommonParams
from .common import PARAM, MM_PARAM, EMB_PARAM, EXCLUDED, TMM_PARAM, Parameter, MM, TMM, Embedding, Linear, call_submodule


def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']

def group_norm(x, num_groups, weight, bias, eps):
    N, C = x.shape[:2]
    G = num_groups
    x = x.reshape(N, G, C // G, *x.shape[2:])
    mean = jnp.mean(x, axis=(2, *range(3, x.ndim)), keepdims=True)
    var = jnp.var(x, axis=(2, *range(3, x.ndim)), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C, *x.shape[3:])
    gamma = weight.reshape(1, C, *([1] * (x.ndim - 2)))
    beta = bias.reshape(1, C, *([1] * (x.ndim - 2)))
    return gamma * x + beta


class LayerNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, eps=1e-5):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        std = jnp.sqrt(var + eps)
        return (x - mean) / std * call_submodule(Parameter, 'weight', common_params) + call_submodule(Parameter, 'bias', common_params)

class GroupNorm(Model):
    @classmethod
    def _forward(cls, common_params, x, num_groups, eps):
        weight = call_submodule(Parameter, 'weight', common_params)
        bias = call_submodule(Parameter, 'bias', common_params)
        N, C = x.shape[:2]
        G = num_groups
        x = x.reshape(N, G, C // G, *x.shape[2:])
        mean = jnp.mean(x, axis=(2, *range(3, x.ndim)), keepdims=True)
        var = jnp.var(x, axis=(2, *range(3, x.ndim)), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        x = x.reshape(N, C, *x.shape[3:])
        gamma = weight.reshape(1, C, *([1] * (x.ndim - 2)))
        beta = bias.reshape(1, C, *([1] * (x.ndim - 2)))
        return gamma * x + beta

class ChannelMixing(Model):
    @classmethod
    def _forward(cls, common_params, x, state, length, new_starts):
        sx = jnp.concatenate([state, x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x
        xk = x + sx * call_submodule(Parameter, 'x_k', common_params)
        k = jnp.square(jax.nn.relu(call_submodule(Linear, 'key', common_params, xk)))
        return call_submodule(Linear, 'value', common_params, k), x[length - 1]


class TimeMixing(Model):
    @classmethod
    def _forward(cls, common_params, x, state, v_first, length, new_starts, H, S, layer_id, inner_loop):
        att = common_params.params
        T, C = x.shape

        sx = jnp.concatenate([state[:1], x[:-1, :]], dtype=x.dtype)
        sx = jnp.where(new_starts[:, None], jnp.zeros_like(sx), sx)
        sx = sx - x

        xr = x + sx * call_submodule(Parameter, 'x_r', common_params)
        xw = x + sx * call_submodule(Parameter, 'x_w', common_params)
        xk = x + sx * call_submodule(Parameter, 'x_k', common_params)
        xv = x + sx * call_submodule(Parameter, 'x_v', common_params)
        xa = x + sx * call_submodule(Parameter, 'x_a', common_params)
        xg = x + sx * call_submodule(Parameter, 'x_g', common_params)

        r = call_submodule(Linear, 'receptance', common_params, xr)
        w = -jax.nn.softplus(-(call_submodule(Parameter, 'w0', common_params) +
                               call_submodule(TMM, 'w2', common_params, jnp.tanh(call_submodule(TMM, 'w1', common_params, xw))))) - 0.5
        k = call_submodule(Linear, 'key', common_params, xk)
        v = call_submodule(Linear, 'value', common_params, xv)

        v_first = jnp.where(layer_id == 0, v, v_first)
        v = jnp.where(layer_id == 0, v, v + (v_first - v) * jax.nn.sigmoid(call_submodule(Parameter, 'v0', common_params) +
                               call_submodule(TMM, 'v2', common_params, (call_submodule(TMM, 'v1', common_params, xv)))))

        a = jax.nn.sigmoid(call_submodule(Parameter, 'a0', common_params) +
                           call_submodule(TMM, 'a2', common_params, (call_submodule(TMM, 'a1', common_params, xa))))
        g = (call_submodule(TMM, 'g2', common_params, jax.nn.sigmoid(call_submodule(TMM, 'g1', common_params, xg))))

        kk = k * call_submodule(Parameter, 'k_k', common_params)
        kk = kk.reshape(T, H, -1)
        kk = kk / jnp.maximum(jnp.linalg.norm(kk, axis=-1, keepdims=True), 1e-12)
        kk = kk.reshape(T, C)
        k = k * (1 + (a-1) * call_submodule(Parameter, 'k_a', common_params))

        state = state.at[0].set(x[length-1])
        s = jnp.reshape(state[1:, :], (H, S, S))

        r, w, k, v, a_i, b_i = tuple([val.reshape(T, H, S) for val in (r, w, k, v, -kk, kk * a)])

        state_new, out = inner_loop(r, w, k, v, a_i, b_i, s, length, new_starts)
        state = state.at[1:].set(state_new.reshape(S, -1))
        x = out.reshape(T, H*S)

        x = call_submodule(GroupNorm, 'ln_x', common_params, x, H, 64e-5)
        x = x + (jnp.sum(r.reshape(1, T, H, -1) * k.reshape(1, T, H, -1)* call_submodule(Parameter, 'r_k', common_params), axis=-1, keepdims=True) * v.reshape(1, T, H, -1)).reshape(T, C)
        x = x * g
        return call_submodule(Linear, 'output', common_params, x), state, v_first
    

class BaseRWKV(LLM):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        import torch
        import re
        w = torch_model
        w['ln0.weight'] = w['blocks.0.ln0.weight']
        w['ln0.bias'] = w['blocks.0.ln0.bias']
        w['blocks.0.att.v0'] = torch.zeros_like(w['blocks.1.att.v0'])
        w['blocks.0.att.v1'] = torch.zeros_like(w['blocks.1.att.v1'])
        w['blocks.0.att.v2'] = torch.zeros_like(w['blocks.1.att.v2'])
        del w['blocks.0.ln0.weight']
        del w['blocks.0.ln0.bias']
        keys = list(w.keys())
        for k in keys:
            if re.match(r'^blocks\.\d+\.ln0\.(weight|bias)$', k):
                print("removing", k)
                del w[k]
            if '.x_' in k or '.k_' in k or '.a0' in k or '.v0' in k or '.w0' in k:
                w[k] = w[k].squeeze()
        return w

    @classmethod
    def get_scan_map(cls, config):
        BS = (0,)
        NS = tuple()
        return {
            'blocks': {
                'att': {'a0': BS, 'a1': BS, 'a2': BS, 'g1': BS, 'g2': BS, 'k_a': BS, 'k_k': BS, 'key': {'weight': BS},
                        'ln_x': {'bias': BS, 'weight': BS}, 'output': {'weight': BS},
                        'r_k': BS, # BS EXCEPTION
                        'receptance': {'weight': BS},
                        'v0': BS, 'v1': BS, 'v2': BS,
                        'value': {'weight': BS},
                        'w0': BS, 'w1': BS, 'w2': BS, 'x_a': BS, 'x_g': BS, 'x_k': BS, 'x_r': BS, 'x_v': BS, 'x_w': BS},
                'ffn': {'key': {'weight': BS}, 'value': {'weight': BS}, 'x_k': BS},
                'ln1': {'bias': BS, 'weight': BS}, 'ln2': {'bias': BS, 'weight': BS}},
            'emb': {'weight': NS},
            'head': {'weight': NS},
            'ln0': {'bias': NS, 'weight': NS},
            'ln_out': {'bias': NS, 'weight': NS}
        }

    @classmethod
    def get_es_map(cls, config):
        LORA = MM_PARAM
        FULL = PARAM
        return {
            'blocks': {
                'att': {'a0': FULL, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': FULL, 'k_k': FULL, 'key': {'weight': LORA},
                        'ln_x': {'bias': FULL, 'weight': FULL}, 'output': {'weight': LORA},
                        'r_k': FULL, # LORA EXCEPTION
                        'receptance': {'weight': LORA},
                        'v0': FULL, 'v1': LORA, 'v2': LORA,
                        'value': {'weight': LORA},
                        'w0': FULL, 'w1': LORA, 'w2': LORA, 'x_a': FULL, 'x_g': FULL, 'x_k': FULL, 'x_r': FULL, 'x_v': FULL, 'x_w': FULL},
                'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': FULL},
                'ln1': {'bias': FULL, 'weight': FULL}, 'ln2': {'bias': FULL, 'weight': FULL}},
            'emb': {'weight': EXCLUDED},
            'head': {'weight': EXCLUDED},
            'ln0': {'bias': FULL, 'weight': FULL},
            'ln_out': {'bias': FULL, 'weight': FULL}
        }

    @classmethod
    def get_quantized_es_map(cls, config):
        LORA = MM_PARAM
        return {
            'blocks': {
                'att': {'a0': EXCLUDED, 'a1': LORA, 'a2': LORA, 'g1': LORA, 'g2': LORA, 'k_a': EXCLUDED, 'k_k': EXCLUDED, 'key': {'weight': LORA},
                        'ln_x': {'bias': EXCLUDED, 'weight': EXCLUDED}, 'output': {'weight': LORA},
                        'r_k': EXCLUDED, # LORA EXCEPTION
                        'receptance': {'weight': LORA},
                        'v0': EXCLUDED, 'v1': LORA, 'v2': LORA,
                        'value': {'weight': LORA},
                        'w0': EXCLUDED, 'w1': LORA, 'w2': LORA, 'x_a': EXCLUDED, 'x_g': EXCLUDED, 'x_k': EXCLUDED, 'x_r': EXCLUDED, 'x_v': EXCLUDED, 'x_w': EXCLUDED},
                'ffn': {'key': {'weight': LORA}, 'value': {'weight': LORA}, 'x_k': EXCLUDED},
                'ln1': {'bias': EXCLUDED, 'weight': EXCLUDED}, 'ln2': {'bias': EXCLUDED, 'weight': EXCLUDED}},
            'emb': {'weight': EXCLUDED},
            'head': {'weight': EXCLUDED},
            'ln0': {'bias': EXCLUDED, 'weight': EXCLUDED},
            'ln_out': {'bias': EXCLUDED, 'weight': EXCLUDED}
        }

    @classmethod
    def get_quantize_map(cls, config):
        FULL = PARAM
        MM = MM_PARAM
        TMM = TMM_PARAM

        return {
            'blocks': {
                'att': {
                    'a0': FULL,
                    'a1': TMM, 'a2': TMM,
                    'g1': TMM, 'g2': TMM,
                    'k_a': FULL,
                    'k_k': FULL,
                    'key': {'weight': MM},
                    'ln_x': {'bias': FULL, 'weight': FULL},
                    'output': {'weight': MM},
                    'r_k': FULL,
                    'receptance': {'weight': MM},
                    'v0': FULL,
                    'v1': TMM, 'v2': TMM,
                    'value': {'weight': MM},
                    'w0': FULL,
                    'w1': TMM, 'w2': TMM,
                    'x_a': FULL,
                    'x_g': FULL,
                    'x_k': FULL,
                    'x_r': FULL,
                    'x_v': FULL,
                    'x_w': FULL,
                },
                'ffn': {
                    'key': {'weight': MM},
                    'value': {'weight': MM},
                    'x_k': FULL,
                },
                'ln1': {'bias': FULL, 'weight': FULL},
                'ln2': {'bias': FULL, 'weight': FULL},
            },
            'emb': {'weight': EXCLUDED},
            'head': {'weight': EXCLUDED},
            'ln0': {'bias': FULL, 'weight': FULL},
            'ln_out': {'bias': FULL, 'weight': FULL},
        }

    @classmethod
    def default_state(cls, params, config):
        n_embd = params['emb']['weight'].shape[1]
        n_layer = params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = params['blocks']['att']['r_k'][0].shape
        return jnp.zeros((n_layer, (2 + head_size), n_embd), dtype=params['emb']['weight'].dtype)

    @classmethod
    def embed(cls, common_params, tokens):
        # TODO: Make this modifiable
        return common_params.params['emb']['weight'][tokens.ravel()]    
    
    @classmethod
    def outhead(cls, common_params, x):
        # TODO: Make this modifiable
        x = call_submodule(LayerNorm, 'ln_out', common_params, x)
        return x @ common_params.params['head']['weight'].T

    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))
        out = jnp.empty_like(r)
        out_s = s

        reset_s = jnp.zeros_like(s)
        for t in range(r.shape[0]):
            s = jax.lax.select(new_starts[t], reset_s, s)
            
            rt = jnp.expand_dims(r[t], 2)
            wt = jnp.expand_dims(w[t], 1)
            kt = jnp.expand_dims(k[t], 1)
            vt = jnp.expand_dims(v[t], 2)
            at = jnp.expand_dims(a[t], 2)
            bt = jnp.expand_dims(b[t], 1)

            sa = s@at
            s = jnp.astype(s * wt + vt @ kt + sa @ bt, s.dtype)
            out = out.at[t].set(jnp.astype((s @ rt).squeeze(2), r.dtype))
            out_s = jax.lax.select(t < length, s, out_s)
        return out_s, out

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        n_layer = common_params.params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = common_params.params['blocks']['att']['r_k'][0].shape
        x = call_submodule(LayerNorm, 'ln0', common_params, x)

        v_first = x

        @partial(jax.checkpoint,
                 policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        def block_loop(y, inputs):
            x, v_first = y
            params_i, es_tree_key_i, state, idx = inputs
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )
            x_new, s, v_first = call_submodule(TimeMixing, 'att', block_i,
                                               call_submodule(LayerNorm, 'ln1', block_i, x),
                                               state[1:], v_first, length, new_starts, n_head, head_size, idx, cls.inner_loop)
            state = state.at[1:].set(s)
            x = x + x_new
            
            x_new, s = call_submodule(ChannelMixing, 'ffn', block_i,
                                      call_submodule(LayerNorm, 'ln2', block_i, x),
                                      state[:1], length, new_starts)
            state = state.at[0].set(s)
            x = x + x_new
            return (x, v_first), state

        (x, _), state = jax.lax.scan(block_loop, (x, v_first), (common_params.params['blocks'], common_params.es_tree_key['blocks'], state, jnp.arange(n_layer)))
        return x, state

    @classmethod
    def features(cls, common_params, x):
        return call_submodule(LayerNorm, 'ln_out', common_params, x)

    @classmethod
    def outhead_from_features(cls, common_params, h):
        return h @ common_params.params['head']['weight'].T

class ScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))
        idxes = jnp.arange(jnp.size(r, axis=0))
        reset_s = jnp.zeros_like(s)
        def scan_loop(inner_states, x):
            out_s, inner_state = inner_states
            r_t, w_t, k_t, v_t, a_t, b_t, t = x
            inner_state = jax.lax.select(new_starts[t], reset_s, inner_state)
            rt = jnp.expand_dims(r_t, 2)
            wt = jnp.expand_dims(w_t, 1)
            kt = jnp.expand_dims(k_t, 1)
            vt = jnp.expand_dims(v_t, 2)
            at = jnp.expand_dims(a_t, 2)
            bt = jnp.expand_dims(b_t, 1)

            sa = inner_state@at
            inner_state = jnp.astype(inner_state * wt + vt @ kt + sa @ bt, inner_state.dtype)
            out_t = jnp.astype((inner_state @ rt).squeeze(2), r_t.dtype)
            out_s = jax.lax.select(t < length, inner_state, out_s)
            return (out_s, inner_state), out_t
        (s, _), out = jax.lax.scan(scan_loop, (s, s), (r, w, k, v, a, b, idxes), unroll=64)
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)

class AssociativeScanRWKV(BaseRWKV):
    @classmethod
    def inner_loop(cls, r, w, k, v, a, b, s, length, new_starts):
        w = jnp.exp(-jnp.exp(w))

        merge_fn = jax.vmap(jax.vmap((lambda wi, ai, bi: jnp.diag(wi) + jnp.outer(ai, bi))))
        v_outer = jax.vmap(jax.vmap(jnp.outer))

        W = merge_fn(w, a, b)
        A = v_outer(v, k)
        all_A = jnp.concatenate((s[None], A), dtype=r.dtype)
        all_W = jnp.concatenate((jnp.zeros_like(W[0:1]), W))
        new_starts = jnp.concatenate((new_starts, jnp.zeros_like(new_starts[:1])))[:, None, None, None]
        all_A = jnp.where(new_starts, jnp.zeros_like(all_A), all_A)
        all_W = jnp.where(new_starts, jnp.zeros_like(all_W), all_W)
        
        def a_scan_loop(elem1, elem2):
            A1, W1 = elem1
            A2, W2 = elem2

            A12 = A2 + A1 @ W2
            W12 = W1 @ W2
            return A12, W12
        scan_s, scan_w = jax.lax.associative_scan(a_scan_loop, (all_A, all_W))

        out = (scan_s[1:] @ jnp.expand_dims(r, -1)).squeeze(-1)
        s = scan_s[length]
        return jnp.astype(s, r.dtype), jnp.astype(out, r.dtype)


class FastRWKV(BaseRWKV):
    @classmethod
    def default_state(cls, params, config):
        n_embd = params['emb']['weight'].shape[1]
        n_layer = params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = params['blocks']['att']['r_k'][0].shape
        return jnp.zeros((n_layer, (2 + head_size), n_embd), dtype=params['emb']['weight'].dtype)

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        n_layer = common_params.params['blocks']['att']['r_k'].shape[0]
        n_head, head_size = common_params.params['blocks']['att']['r_k'][0].shape
        x = call_submodule(LayerNorm, 'ln0', common_params, x)

        v_first = x
        for i in range(n_layer):
            params_i = jax.tree.map(lambda a: a[i], common_params.params['blocks'])
            es_tree_key_i = jax.tree.map(lambda a: a[i], common_params.es_tree_key['blocks'])
            state_i = state[i]
            idx = i
            block_i = common_params._replace(
                params=params_i,
                es_tree_key=es_tree_key_i
            )
            x_new, s, v_first = call_submodule(TimeMixing, 'att', block_i,
                                               call_submodule(LayerNorm, 'ln1', block_i, x),
                                               state_i[1:], v_first, length, new_starts, n_head, head_size, idx, cls.inner_loop)
            state_i = state_i.at[1:].set(s)
            x = x + x_new
            
            x_new, s = call_submodule(ChannelMixing, 'ffn', block_i,
                                      call_submodule(LayerNorm, 'ln2', block_i, x),
                                      state_i[:1], length, new_starts)
            state_i = state_i.at[0].set(s)
            x = x + x_new
            state = state.at[i].set(state_i)

        return x, state