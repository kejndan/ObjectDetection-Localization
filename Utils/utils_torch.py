import torch

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def load_model(load_path, model, cfg, optimizer=None):
    checkpoint = torch.load(load_path)
    load_state_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if key in load_state_dict:
            model_state_dict[key] = load_state_dict[key]
    model.load_state_dict(model_state_dict)
    start_epoch = checkpoint['epoch'] + 1
    best_score = checkpoint['best_score']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['opt'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(cfg.device)
    return model, optimizer, start_epoch, best_score

