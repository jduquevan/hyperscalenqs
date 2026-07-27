import jax
import jax.numpy as jnp
from .base_model import Model, CommonInit, CommonParams

def get_int_component(x):
    parts = x.split('.')
    return tuple([int(p) for p in parts if p.isdigit()])

class LLM(Model):
    @classmethod
    def transform_torch_model(cls, torch_model, dtype=jnp.bfloat16):
        return torch_model

    @classmethod
    def transform_config(cls, config):
        return config

    @classmethod
    def get_scan_map(cls, config):
        raise NotImplementedError("Scan map is not implemented")
    
    @classmethod
    def get_es_map(cls, config):
        raise NotImplementedError("Scan map is not implemented")
    
    @classmethod
    def load_from_torch(cls, torch_model, config, dtype=jnp.bfloat16):
        """
        Loads from torch (cpu) model and outputs a jax (cpu) model
        """
        w = cls.transform_torch_model(torch_model, dtype=dtype)
        del torch_model
        print("transformed torch model")
        cfg = cls.transform_config(config)
        for k in w.keys():
            # w[k] = jnp.array(w[k].float().numpy(), dtype=dtype, device=jax.local_devices(backend="cpu")[0])
            cpu = jax.local_devices(backend="cpu")[0]
            with jax.default_device(cpu):
                w[k] = jnp.asarray(w[k].float().cpu().numpy(), dtype=dtype)
        print("got w in jax")

        ans = {}
        ans['blocks'] = {}
        for k in sorted(w.keys(), key=get_int_component):
            parts = k.split('.')
            last = parts.pop()
            here = ans
            add_list = False
            for p in parts:
                if p.isdigit():
                    add_list = True
                else:
                    if p not in here:
                        here[p] = {}
                    here = here[p]
            if not add_list:
                here[last] = w[k]
            else:
                if last not in here:
                    here[last] = [w[k]]
                else:
                    here[last].append(w[k])
        print("converting final params")
        ans = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, list) else x, ans, is_leaf=lambda x: isinstance(x, list))
        print("got converted")
        return CommonInit(
            cfg,
            ans,
            cls.get_scan_map(cfg),
            cls.get_es_map(cfg)
        )

    @classmethod
    def randomize_weights(cls, key, n_layer, n_embd, vocab_size, config, dtype):
        raise NotImplementedError("Randomize Weights is not implemented")

    @classmethod
    def default_state(cls, params, config):
        raise NotImplementedError("Default State is not implemented")

    @classmethod
    def embed(cls, common_params, tokens):
        raise NotImplementedError("Embedding function is not implemented")
    
    @classmethod
    def outhead(cls, common_params, x):
        raise NotImplementedError("Out head is not implemented")

    @classmethod
    def forward_seq(cls, common_params, x, state, length, new_starts):
        raise NotImplementedError("Forward sequence is not implemented")

    @classmethod
    def _forward(cls, common_params, tokens, state, length=None, new_starts=None):
        """
        Forward pass on a single stream of tokens
        """
        tokens = jnp.array(tokens)
        x = cls.embed(common_params, tokens)
        T, D = x.shape
        if length is None:
            length = T
        if new_starts is None:
            new_starts = jnp.zeros((T,), dtype=jnp.bool)
        x, state = cls.forward_seq(common_params, x, state, length, new_starts)
        x = cls.outhead(common_params, x)
        return x, state