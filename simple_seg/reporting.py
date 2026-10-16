"""Human-readable experiment facts taken from resolved configs, not model names."""
from collections.abc import Mapping
from copy import deepcopy


def describe_config(cfg):
    model = cfg.get('model', {})
    model = model if isinstance(model, Mapping) else {}
    losses = []
    def visit(node, path):
        if isinstance(node, Mapping):
            for key, value in node.items():
                target = path + '.' + key if path else key
                if key.startswith('loss') and isinstance(value, (Mapping, list, tuple)):
                    items = value if isinstance(value, (list, tuple)) else [value]
                    for index, item in enumerate(items):
                        if isinstance(item, Mapping) and 'type' in item:
                            losses.append(dict(path=target + ('[{}]'.format(index) if len(items) > 1 else ''),
                                               config=deepcopy(dict(item))))
                else:
                    visit(value, target)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                visit(value, path + '[{}]'.format(index))
    visit(model.get('decode_head', {}), 'decode_head')
    visit(model.get('auxiliary_head', {}), 'auxiliary_head')
    wrapper = cfg.get('optim_wrapper', {})
    wrapper = wrapper if isinstance(wrapper, Mapping) else {}
    optimizer = wrapper.get('optimizer', {})
    return dict(model_type=model.get('type'), losses=losses,
                optimizer=deepcopy(dict(optimizer)) if isinstance(optimizer, Mapping) else str(type(optimizer)),
                paramwise_cfg=deepcopy(wrapper.get('paramwise_cfg', {})),
                optimizer_constructor=wrapper.get('constructor'),
                scheduler=deepcopy(cfg.get('param_scheduler', [])))
