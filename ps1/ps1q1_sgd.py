import numpy as np
import matplotlib.pyplot as plt
import time
data = np.loadtxt('perceptron.data',delimiter=',')
x = data[:, 0:4]
y = data[:, 4:]
step_size_losses = {}
for step_size in [0.01, 1, 10.99]:
    w = np.zeros((1,4))
    b = 0
    
    epoch = 1
    losses = []
    i = 0
    while True:
        # Predict
        z = x.dot(w.T) + b
        y_pred = np.sign(z)

        # SGD on i-th data point
        mask = (-1 * y[i] * y_pred[i]) >= 0
        dw = -1 * (y[i] * x[i] * mask) / len(y[i])
        db = -1 * (y[i] * mask) / len(y[i])
        
        w = w - step_size * dw
        b = b - step_size * db

        # Loss
        i += 1
        if i % 1000 == 0:
            if epoch in [1,2,3]:
                print('Epoch', epoch, 'Weights', w, 'Bias', b)
            
            z = x.dot(w.T) + b
            loss_fn = -1 * y * z
            loss_fn = loss_fn * (loss_fn > 0)
            loss = np.sum(loss_fn)
            losses.append(loss)
            if loss == 0:
                break
            print("Epoch", epoch, "Loss", loss)
            epoch += 1
            i = 0
            print("----------------------------------------------------")
    step_size_losses[step_size] = losses
    print('Final Epoch', epoch, 'Step size', step_size, 'Weights', w, 'Bias', b)

for key, value in step_size_losses.items():
    plt.plot(value, label='Step size ' + str(key))

plt.title('Iteration vs Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()