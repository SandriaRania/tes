import os
import numpy as np, yaml
from typing import Any, IO

def sample_discrete(pvals):
    return np.random.multinomial(n=1, pvals=pvals).tolist().index(1)

def read_cfg(fpath):
    with open(fpath, 'r') as f:
        cfg = yaml.load(f, Loader=YAMLCustomLoader)
    return cfg

################################################### yaml ###########################################
# https://stackoverflow.com/questions/33490870/parsing-yaml-in-python-detect-duplicated-keys
# https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
# https://gist.github.com/joshbode/569627ced3076931b02f
class YAMLCustomLoader(yaml.SafeLoader):
    def __init__(self, stream: IO) -> None:
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)
    
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping, f'duplicated key: {key}'
            mapping.append(key)
        return super().construct_mapping(node, deep)

def yaml_construct_include(loader: YAMLCustomLoader, node: yaml.Node) -> Any:
    fpath = os.path.expanduser(os.path.join(loader._root, loader.construct_scalar(node)))
    if not os.path.exists(fpath):
        raise NotImplementedError
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            return yaml.load(f, YAMLCustomLoader)
    return None

yaml.add_constructor('!include', yaml_construct_include, YAMLCustomLoader)
