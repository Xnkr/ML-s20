import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('perceptron.data',delimiter=',')
x = data[:, 0:4]
y = data[:, 4:]
print('Standard Gradient Descent')
step_size_losses = {}
for step_size in [0.01, 1, 10.99]:
    w = np.zeros((1,4))
    b = 0
    losses = []
    i = 1
    while True:
        # Predict
        z = x.dot(w.T) + b
        y_pred = np.sign(z)

        # Gradient descent
        mask = (-1 * y * y_pred) >= 0
        dw = -1 * np.sum(y * x * mask, axis=0)
        db = -1 * np.sum(y * mask)

        if np.all(dw == 0) and db == 0:
            break

        w = w - step_size * dw
        b = b - step_size * db
        
        if i in [1,2,3]:
            print('Iteration', i, 'Weights', w, 'Bias', b)

        # Loss
        z = x.dot(w.T) + b
        loss_fn = -1 * y * z
        loss_fn = loss_fn * (loss_fn > 0)
        loss = np.sum(loss_fn)
        print("Iteration", i, "Loss", loss)
        losses.append(loss)
        i += 1
        print("----------------------------------------------------")
    step_size_losses[step_size] = losses
    print('Final Iteration', i, 'Step size', step_size, 'Weights', w, 'Bias', b)

for key, value in step_size_losses.items():
    plt.plot(value, label='Step size ' + str(key))

plt.title('Iteration vs Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()