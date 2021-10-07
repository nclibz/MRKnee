class Cfg:
    def __init__(self, **kwargs):
        self.objs = kwargs
        self.all_cfgs = dict()
        for cfg in self.objs.values():
            self.all_cfgs.update(cfg.__dict__)

    def get_cfg(self):
        return {k: v for k, v in self.all_cfgs.items() if not k.startswith("_")}
