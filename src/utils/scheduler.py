class MyScheduler():
    
    def __init__(self, base_opt, init_lr, d_model, n_warmup):
        self.base_opt = base_opt
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup = n_warmup
        self.n_step = 0
        
    def zero_grad(self):
        self.base_opt.zero_grad()
        
    def state_dict(self):
        return {
            'base_opt': self.base_opt(),
            'init_lr': self.init_lr,
            'd_model': self.d_model,
            'n_warmup': self.n_warmup,
            'n_step': self.n_step
        }
    
    def load_state_dict(self, state_dict):
        self.base_opt.load_state_dict(state_dict['base_opt'])
        self.init_lr = state_dict['init_lr']
        self.d_model = state_dict['d_model']
        self.n_warmup = state_dict['n_warmup']
        self.n_step = state_dict['n_step']
        
    def step_and_update_lr(self):
        self._update_lr()
        self.base_opt.step()
        
    def _update_lr(self):
        self.n_step += 1
        lr = self.init_lr * self._get_lr_scale()
        
        for params in self.base_opt.param_groups:
            params['lr'] = lr
            
    def _get_lr_scale(self):
        d_model = self.d_model
        n_warmup = self.n_warmup
        n_step = self.n_step
        
        a = (d_model ** (-0.5))
        b = min(n_step ** (-0.5), n_step * n_warmup ** (-1.5))
        return a*b 