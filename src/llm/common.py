import jax
import jax.numpy as jnp

from .base_model import Model, CommonInit, CommonParams

PARAM = 0
MM_PARAM = 1
EMB_PARAM = 2
EXCLUDED = 3
TMM_PARAM = 5

def layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std

ACTIVATIONS = {
    'relu': jax.nn.relu,
    'silu': jax.nn.silu,
    'pqn': lambda x: jax.nn.relu(layer_norm(x))
}

def recursive_scan_split(param, base_key, scan_tuple):
    # scan_tuple = tuple() implies no split
    if len(scan_tuple) == 0:
        return base_key
    # otherwise, it is (0, 1, ...)
    split_keys = jax.random.split(base_key, param.shape[scan_tuple[0]])
    return jax.vmap(recursive_scan_split, in_axes=(None, 0, None))(param, split_keys, scan_tuple[1:])

def simple_es_tree_key(params, base_key, scan_map):
    vals, treedef = jax.tree.flatten(params)
    all_keys = jax.random.split(base_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    return jax.tree.map(recursive_scan_split, params, partial_key_tree, scan_map)

def merge_inits(**kwargs):
    params = {}
    frozen_params = {}
    scan_map = {}
    es_map = {}
    for k in kwargs:
        params[k] = kwargs[k].params #k_params
        scan_map[k] = kwargs[k].scan_map #k_scan_map
        es_map[k] = kwargs[k].es_map #k_es_map
        if kwargs[k].frozen_params is not None:
            frozen_params[k] = kwargs[k].frozen_params
    if not frozen_params:
        frozen_params = None

    return CommonInit(frozen_params, params, scan_map, es_map)

def merge_frozen(common, **kwargs):
    new_frozen_params = common.frozen_params or {}
    new_frozen_params = new_frozen_params | kwargs
    return common._replace(frozen_params=new_frozen_params)

def call_submodule(cls, name, common_params, *args, **kwargs):
    sub_common_params = common_params._replace(
        frozen_params=common_params.frozen_params[name] if common_params.frozen_params and name in common_params.frozen_params else None,
        params=common_params.params[name],
        es_tree_key=common_params.es_tree_key[name]
    )
    return cls._forward(sub_common_params, *args, **kwargs)

class Parameter(Model):
    @classmethod
    def rand_init(cls, key, shape, scale, raw_value, dtype, *args, **kwargs):
        if raw_value is not None:
            params = raw_value.astype(dtype=dtype)
        else:
            params = (jax.random.normal(key, shape) * scale).astype(dtype=dtype)
        
        frozen_params = None
        scan_map = ()
        es_map = PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        return common_params.noiser.get_noisy_standard(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo)

class MM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = (jax.random.normal(key, (out_dim, in_dim)) * scale).astype(dtype=dtype)
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_mm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class TMM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = MM_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_Tmm(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class Embedding(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, *args, **kwargs):
        scale = 1 / jnp.sqrt(in_dim)
        params = jax.random.normal(key, (in_dim, out_dim), dtype=dtype) * scale
        frozen_params = None
        scan_map = ()
        es_map = EMB_PARAM
        return CommonInit(frozen_params, params, scan_map, es_map)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        return common_params.noiser.do_emb(common_params.frozen_noiser_params, common_params.noiser_params, common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, use_bias, dtype, *args, **kwargs):
        if use_bias:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
                bias=Parameter.rand_init(key, None, None, jnp.zeros(out_dim, dtype=dtype), dtype)
            )
        else:
            return merge_inits(
                weight=MM.rand_init(key, in_dim, out_dim, dtype),
            )

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        ans = call_submodule(MM, 'weight', common_params, x)
        if "bias" in common_params.params:
            ans += call_submodule(Parameter, 'bias', common_params)
        return ans
            
class MLP(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dims, use_bias, activation, dtype, *args, **kwargs):
        input_dims = [in_dim] + list(hidden_dims)
        output_dims = list(hidden_dims) + [out_dim]

        all_keys = jax.random.split(key, len(input_dims))

        merged_params = merge_inits(**{str(t): Linear.rand_init(all_keys[t], input_dims[t], output_dims[t], use_bias, dtype) for t in range(len(input_dims))})
        return merge_frozen(merged_params, activation=activation)

    @classmethod
    def _forward(cls, common_params, x, *args, **kwargs):
        num_blocks = len(common_params.params)
        for t in range(num_blocks):
            x = call_submodule(Linear, str(t), common_params, x)
            if t != num_blocks - 1:
                x = ACTIVATIONS[common_params.frozen_params['activation']](x)
        return x