disp("Standard Gradient Descent");
percep = load('perceptron.data');
X = percep(:, 1:4);
sizeX = size(X);
m = sizeX(1);
n = sizeX(2);
Y = percep(:, 5);
w = zeros(1,4);
b = 0;
step_size = 2;
loss = 1;
iter = 0;
while loss ~= 0
    z = X*w' + b;
    a = arrayfun(@step, z);
    iter = iter + 1;
    if iter == 1 || iter == 2 || iter == 3
        display = ['Iteration ', num2str(iter), ' Weights: ', mat2str(w), ' Bias: ', num2str(b)];
        disp(display);
    end
    loss = 0;
    for j = 1:m
        loss = loss + max([0, -(Y(j)*a(j))]);
    end
    if loss == 0
        break
    end
    dw = zeros(1,4);
    db = 0;
    for i = 1:m
       if ((-Y(i) * (w * X(i, :)' + b)) >= 0)
           dw = dw + (-Y(i) * X(i, :));
           db = db + (-Y(i));
       end
    end
    w = w - (step_size * dw);
    b = b - (step_size * db);
end
display = ['Iteration ', num2str(iter), ' Weights: ', mat2str(w), ' Bias: ', num2str(b)];
disp(display);
function i = step(x)
    if x >= 0
        i = 1;
    else
        i = -1;
    end
end