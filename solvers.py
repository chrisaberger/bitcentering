import utils
import numpy as np

def sgd_baseline(d, model):
    for epoch in range(0, d.num_epochs):
        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)
            cost += model.forward(x, y)
            model.backward()
            model.step()

        predY = model.predict(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))

def svrg_baseline(d, model):
    w_tilde = None
    g_tilde = None
    #model.lr *= 2
    for epoch in range(0, d.num_epochs):
        if epoch % d.T == 0:
            w_tilde = np.copy(model.lin_layer.weight.data())
            cost  = model.forward(d.x_train, d.y_train)
            model.backward()
            g_tilde = np.copy(model.lin_layer.weight.offset_grad)

        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)

            w_offset = np.copy(model.lin_layer.weight.offset)
            np.copyto(model.lin_layer.weight.offset, w_tilde)

            model.forward(x, y)
            model.backward()
            w_tilde_grad = np.copy(model.lin_layer.weight.offset_grad)

            np.copyto(model.lin_layer.weight.offset, w_offset)
            cost += model.forward(x, y)
            model.backward()
            model.step_svrg(w_tilde_grad, g_tilde)

        predY = model.predict(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))

def lp_svrg_baseline(d, model):
    w_tilde = None
    g_tilde = None

    for epoch in range(0, d.num_epochs):
        if epoch % d.T == 0:
            w_tilde = np.copy(model.lin_layer.weight.data())
            cost  = model.forward(d.x_train, d.y_train)
            model.backward()
            g_tilde = np.copy(model.lin_layer.weight.offset_grad)

        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)

            w_offset = np.copy(model.lin_layer.weight.offset)
            np.copyto(model.lin_layer.weight.offset, w_tilde)

            model.forward_lp(x, y)
            model.backward_lp()
            w_tilde_grad = np.copy(model.lin_layer.weight.offset_grad)

            np.copyto(model.lin_layer.weight.offset, w_offset)
            cost += model.forward_lp(x, y)
            model.backward_lp()
            model.step_svrg_lp(w_tilde_grad, g_tilde)

        predY = model.predict(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))

def lp_sgd_baseline(d, model):
    for epoch in range(0, d.num_epochs):
        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)
            cost += model.forward_lp(x, y)
            model.backward_lp()
            model.step()

        predY = model.predict(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))

def sgd_bitcentering(d, model):
    for epoch in range(0, d.num_epochs):
        if epoch % d.T == 0:
            model.recenter()
            # Cache the results.
            for batch_index in range(0, d.num_batches):
                x, y = d.get_data(batch_index)
                model.forward_store(x, y, batch_index)
                model.backward_store(batch_index)

        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)
            cost += model.forward_inner(x, y, batch_index)
            model.backward_inner(batch_index)
            model.step_inner()

        predY = model.predict_inner(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))


def svrg_bitcentering(d, model):
    g_tilde = None
    cost = 0
    for epoch in range(0, d.num_epochs):
        if epoch % d.T == 0:
            model.recenter()
            cost  = model.forward(d.x_train, d.y_train)
            model.backward()
            g_tilde = np.copy(model.lin_layer.weight.offset_grad)

            # Cache the results.
            for batch_index in range(0, d.num_batches):
                x, y = d.get_data(batch_index)
                model.forward_store(x, y, batch_index)
                model.backward_store(batch_index)

        cost = 0
        for batch_index in range(0, d.num_batches):
            x, y = d.get_data(batch_index)

            cost += model.forward_inner(x, y, batch_index)
            model.backward_inner(batch_index)
            model.step_svrg_inner(g_tilde, batch_index)

        predY = model.predict_inner(d.x_test)
        utils.print_info(epoch, 
                         cost/d.num_batches, 
                         100*np.mean(predY == d.y_test))