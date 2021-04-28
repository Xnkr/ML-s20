disp("Stochastic Gradient Descent");
percep = load('perceptron.data');
X = percep(:, 1);
sizeX = size(X);
m = sizeX(1);
n = sizeX(2);
Y = percep(:, 5);
w = zeros(1,1);
b = 0;
step_size = 1;
loss = 1;
iter = 0;
while loss ~= 0
    z = X*w' + b;
    a = arrayfun(@step, z);
    iter = iter + 1;
    loss = 0;
    for j = 1:m
        loss = loss + max([0, -(Y(j)*a(j))]);
    end
    if iter == 1 || iter == 2 || iter == 3 || mod(iter, 100) == 0
        display = ['Iteration ', num2str(iter), ' Weights: ', mat2str(w), ' Bias: ', num2str(b)];
        disp(display);
        disp(loss);
    end
    if loss == 0
        break
    end
    dw = zeros(1,1);
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
scatter(X,Y);
hold on;
x0 = min(X) ; x1 = max(X) ;
xi = linspace(x0,x1) ;
yi = xi*w'+b;
plot(xi,yi);

function i = step(x)
    if x >= 0
        i = 1;
    else
        i = -1;
    end
end