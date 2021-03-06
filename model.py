from linear import Linear
from loss import *
from splittensor import SplitTensor

class LogisticRegression:
    def __init__(self, 
                 n_samples, 
                 batch_size, 
                 n_bits, 
                 fwd_scale_factor,
                 bck_scale_factor,
                 loss_scale_factor,
                 in_features, 
                 out_features, 
                 lr):
        self.lin_layer = Linear(n_samples=n_samples, 
                           batch_size=batch_size, 
                           n_bits=n_bits,
                           fwd_scale_factor=fwd_scale_factor,
                           bck_scale_factor=bck_scale_factor, 
                           in_features=in_features, 
                           out_features=out_features)
        self.loss_layer = CrossEntropy(n_samples, out_features, batch_size, n_bits, loss_scale_factor)
        self.lr = lr
        self.fwd_scale_factor = fwd_scale_factor
        self.bck_scale_factor = bck_scale_factor
        self.loss_scale_factor = loss_scale_factor

    def predict(self, x):
        fwd = self.lin_layer.forward(x, train=False)
        return fwd.argmax(axis=1)

    def recenter(self):
        self.lin_layer.recenter()

    ######################### Baseline Methods #################################
    def forward(self, x, y):
        fwd = self.lin_layer.forward(x)
        return self.loss_layer.forward(fwd, y)

    def backward(self):
        self.lin_layer.backward(self.loss_layer.backward())

    def step(self):
        self.lin_layer.step(self.lr)
    ############################################################################

    ######################### LP Baseline Methods ##############################
    def forward_lp(self, x, y):
        fwd = self.lin_layer.forward_lp(x)
        return self.loss_layer.forward_interp(fwd, y)

    def backward_lp(self):
        self.lin_layer.backward_lp(self.loss_layer.backward_lp())

    def step_lp(self):
        self.lin_layer.step_lp(self.lr)
    ############################################################################

    ########################### Outer Methods ##################################
    def forward_store(self, x, y, batch_index):
        fwd = self.lin_layer.forward_store(x, batch_index)
        return self.loss_layer.forward_store(fwd, y, batch_index)

    def backward_store(self, batch_index):
        self.lin_layer.backward_store(self.loss_layer.backward(), batch_index)
    ############################################################################

    ########################### Inner Methods ##################################
    def forward_inner(self, x, y, batch_index):
        fwd = self.lin_layer.forward_inner(SplitTensor(x), batch_index)
        return self.loss_layer.forward_interp(fwd, y)

    def backward_inner(self, batch_index):
        loss_bck = self.loss_layer.backward_inner(batch_index)
        self.lin_layer.backward_inner(loss_bck, batch_index)

    def step_inner(self):
        self.lin_layer.step_inner(self.lr)

    def predict_inner(self, x):
        fwd = np.dot(x, self.lin_layer.weight.data().T)
        return fwd.argmax(axis=1)
    ############################################################################

    def step_svrg(self, w_tilde_grad, g_tilde):
        self.lin_layer.step_svrg(w_tilde_grad, g_tilde, self.lr)

    def step_svrg_inner(self, g_tilde, batch_index):
        self.lin_layer.step_svrg_inner(g_tilde, self.lr, batch_index)
    
    def step_svrg_lp(self, w_tilde_grad, g_tilde):
        self.lin_layer.step_svrg_lp(g_tilde, w_tilde_grad, self.lr)
