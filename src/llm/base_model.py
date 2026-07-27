from typing import NamedTuple

class CommonInit(NamedTuple):
    frozen_params: any
    params: any
    scan_map: any
    es_map: any

class CommonParams(NamedTuple):
    noiser: any
    frozen_noiser_params: any
    noiser_params: any
    frozen_params: any
    params: any
    es_tree_key: any
    iterinfo: any

class Model:
    @classmethod
    def rand_init(cls, key, *args, **kwargs):
        """
        Initialize model

        returns frozen_params, params, scan_map, es_map as CommonInit
        """
        raise NotImplementedError("Randomize Weights is not implemented")

    @classmethod
    def forward(cls,
                noiser, frozen_noiser_params, noiser_params,
                frozen_params, params, es_tree_key, iterinfo, *args, **kwargs):
        """
        Forward pass of model

        returns just the output
        """
        return cls._forward(CommonParams(noiser, frozen_noiser_params, noiser_params, frozen_params, params, es_tree_key, iterinfo), *args, **kwargs)

    @classmethod
    def _forward(cls, common_params, *args, **kwargs):
        raise NotImplementedError("Forward pass is not implemented")