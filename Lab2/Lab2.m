%clc;
%clear;

global DERIVATIVE_TYPE_DIFF;
global DERIVATIVE_TYPE_LEFT;
global DERIVATIVE_TYPE_RIGHT;
global DERIVATIVE_TYPE_CENTRAL;
global derivative_type;
global ddf;
global df;
global df1;
global df2;

global DERIVATIVE_PRECISION;
global FIRST_DERIVATIVE_COST;
global SECOND_DERIVATIVE_COST;
global GRADIENT_COST;

DERIVATIVE_TYPE_DIFF    = 0;
DERIVATIVE_TYPE_LEFT    = 1;
DERIVATIVE_TYPE_RIGHT   = 2;
DERIVATIVE_TYPE_CENTRAL = 3;

derivative_type = DERIVATIVE_TYPE_CENTRAL;

if derivative_type == DERIVATIVE_TYPE_DIFF
    FIRST_DERIVATIVE_COST = 1;
    SECOND_DERIVATIVE_COST = 1;
else
    FIRST_DERIVATIVE_COST = 2;
    SECOND_DERIVATIVE_COST = 3;
end

GRADIENT_COST = 2;

function [derivative_val] = calculate_left_derivative(f, x, h)
    derivative_val = (f(x) - f(x - h)) / h;
end

function [derivative_val] = calculate_right_derivative(f, x, h)
    derivative_val = (f(x + h) - f(x)) / h;
end

function [derivative_val] = calculate_central_derivative(f, x, h)
    derivative_val = (f(x + h) - f(x - h)) / (h * 2);
end

function [derivative_val] = calculate_second_derivative(f, x, h)
    derivative_val = (f(x + h) - (2*f(x)) + f(x - h)) / h^2;
end

function [derivative_val] = first_derivative(func, arg)
    global DERIVATIVE_TYPE_DIFF;
    global DERIVATIVE_TYPE_LEFT;
    global DERIVATIVE_TYPE_RIGHT;
    global DERIVATIVE_TYPE_CENTRAL;
    global derivative_type;
    global DERIVATIVE_PRECISION;
    global df;
    switch derivative_type
        case DERIVATIVE_TYPE_DIFF
            derivative_val = double(df(arg));
        case DERIVATIVE_TYPE_LEFT
            derivative_val = calculate_left_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_RIGHT
            derivative_val = calculate_right_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_CENTRAL
            derivative_val = calculate_central_derivative(func, arg, DERIVATIVE_PRECISION);
    end
end

function [derivative_val] = second_derivative(func, arg)
    global DERIVATIVE_TYPE_DIFF;
    global DERIVATIVE_TYPE_LEFT;
    global DERIVATIVE_TYPE_RIGHT;
    global DERIVATIVE_TYPE_CENTRAL;
    global derivative_type;
    global DERIVATIVE_PRECISION;
    global ddf;
    switch derivative_type
        case DERIVATIVE_TYPE_DIFF
            derivative_val = double(ddf(arg));
        case DERIVATIVE_TYPE_LEFT
            derivative_val = calculate_second_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_RIGHT
            derivative_val = calculate_second_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_CENTRAL
            derivative_val = calculate_second_derivative(func, arg, DERIVATIVE_PRECISION);
    end
end

function [grad] = func_gradient(f, x)
    global df1;
    global df2;
    grad = [double(df1(x(1), x(2))), double(df2(x(1), x(2)))];
end

function [xmin, fmin, func_calculations] = digitwise(f, start, stop, precision, k)
    delta = 1;
    x = start;
    direction = 1;
    y = f(x);
    func_calculations = 1;
    while delta > precision
        delta = delta / k;
        yprev = y;
        func_calculations = func_calculations + 1;
        x = x + delta*direction;
        y = f(x);
        func_calculations = func_calculations + 1;
        while y < yprev
            yprev = y;
            xprev = x;
            x = x + delta*direction;
            y = f(x);
            func_calculations = func_calculations + 1;
        end
        direction = direction * -1;
    end
    fmin = yprev;
    xmin = xprev;
end

function [xmin, fmin, func_calculations] = steepest_descent(f, x0, epsilon)
    global GRADIENT_COST;
    % Метод наискорейшего спуска
    x = x0;
    func_calculations = 0;
    max_iter = 1000;
    func = @(x) f(x(1), x(2));
    
    for iter = 1:max_iter
        % Вычисление градиента
        %[grad, deriv_evals] = numerical_gradient(f, x);
        grad = func_gradient(f,x);
        func_calculations = func_calculations + GRADIENT_COST;
        % Критерий остановки
        if norm(grad) < epsilon
            break;
        end
      
        % Поиск оптимального шага
        [alpha, Fmin, calcs] = digitwise(@(a) func(x - a * grad), 0, 1, epsilon, 4);
        func_calculations = func_calculations + calcs;
        
        x_new = x - alpha * grad;
        x = x_new;
    end
    
    xmin = x;
    fmin = func(x);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations] = conjugate_gradient(f, x0, epsilon)
    global GRADIENT_COST;
    % Метод сопряженных градиентов (Флетчера-Ривса)
    x = x0;
    func_calculations = 0;
    max_iter = 1000;
    func = @(x) f(x(1), x(2));
    
    grad = func_gradient(f,x);
    func_calculations = func_calculations + GRADIENT_COST;
    d = -grad;
    
    for iter = 1:max_iter
        % Поиск оптимального шага
        [alpha, Fmin, calcs] = digitwise(@(a) func(x + a * d), 0, 1, epsilon, 4);
        func_calculations = func_calculations + calcs;

        x = x + alpha * d;

        grad_prev = grad;
        grad = func_gradient(f,x);
        func_calculations = func_calculations + GRADIENT_COST;

        beta = norm(grad)^2 / norm(grad_prev)^2;
        d = -grad + beta * d;

        if norm(d) < epsilon
            break;
        end
        
        
    end
    xmin = x;
    fmin = func(x);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations] = Newton(func, start, target_precision)
    global FIRST_DERIVATIVE_COST;
    global SECOND_DERIVATIVE_COST;
    func_calculations = 0;
    xmin = start;
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    func_calculations = func_calculations + FIRST_DERIVATIVE_COST + SECOND_DERIVATIVE_COST;
    step = abs(dfmin/ddfmin);
    while abs(dfmin) > target_precision
        xmin = xmin - dfmin/ddfmin;
        prevstep = step;
        step = abs(dfmin/ddfmin);
        if (step - prevstep) > 0
            xmin = Inf;
            break
        end 
        dfmin = first_derivative(func, xmin);
        ddfmin = second_derivative(func, xmin);
        func_calculations = func_calculations + FIRST_DERIVATIVE_COST + SECOND_DERIVATIVE_COST;
    end
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end


function plotSingleFunction(f, start, stop)
    % Create x values symmetric around zero
    x = linspace(start, stop, 1000);
    y = f(x);
    step = 0.2;
    % Create figure and plot
    figure;
    plot(x, y, 'b-', 'LineWidth', 1.5);
    
    % Axis labels with formatting
    xlabel('x', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('f(x)', 'FontSize', 18, 'FontWeight', 'bold');
    functext = func2str(f);
    functext = erase(functext, '.');
    titletext = sprintf('Graph of f(x) = %s on interval [%d:%d]', functext(5:end), start, stop);
    title(titletext, 'FontSize', 14);
    
    % Set axis limits to include origin
    xlim([start-1, stop+1]);
    ylim([min(y)-1, max(y)+0.5]);
    
    % Configure axis divisions (price delenia)
    xticks(start-1:step:stop+1);
    yticks(floor(min(y)-1):step:ceil(max(y)+0.5));
    
    % Emphasize axes
    grid on;
    ax = gca;
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.3;
    
    % Highlight the coordinate origin
    hold on;
    plot(0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'black');
    
    % Emphasize x and y axes with thicker lines
    xl = xlim;
    yl = ylim;
    line(xl, [0, 0], 'Color', 'k', 'LineWidth', 1.2); % x-axis
    line([0, 0], yl, 'Color', 'k', 'LineWidth', 1.2); % y-axis
    
    hold off;
end

function plotTableFunctionsLogLog(funcNames, tableData, xLimits, yLimits)
    % tableData: table where first column is x-values, subsequent columns are y-values
    % xLimits: [xmin, xmax] for x-axis limits
    % yLimits: [ymin, ymax] for y-axis limits
    
    % Extract x and y data
    x = tableData(:, 1); % First column as x-values
    
    figure;
    hold on;
    
    % Plot each function (each column after the first)
    numFunctions = size(tableData, 2) - 1;
    colors = lines(numFunctions); % Different colors for each function
    styles = ['o', '+', '*', 'x', 'square', '^', 'v', ">", "<", "pentagram", "hexagram"];
    
    for i = 1:numFunctions
        y = tableData(:, i+1); % Subsequent columns as y-values
        
        % Create log-log plot using set(gca) method :cite[2]
        plot(x, y, sprintf('-%s', styles(i)), 'LineWidth', 1.5, 'Color', colors(i, :), ...
             'DisplayName', sprintf('%s', funcNames(i+1)));
    end
    
    % Set logarithmic scale for both axes
    %xscale log;
    %yscale log;
    
    % Set axis limits if provided
    if exist('xLimits', 'var') && ~isempty(xLimits)
        xlim(xLimits);
    end
    if exist('yLimits', 'var') && ~isempty(yLimits)
        ylim(yLimits);
    end
    
    % Labels and formatting
    xlabel('Precision', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('Function calculations', 'FontSize', 18, 'FontWeight', 'bold');
    title('Algorithms speed', 'FontSize', 14);
    
    legend('show', 'Location', 'best');
    grid on;
    
    hold off;
end

target_precision = 0.000001;
DERIVATIVE_PRECISION = target_precision;
x0 = [1, -1];
a = 1;
f = @(x1, x2) x1.^2 + a * x2.^2;
%f = @(x) x(1).^2 + a * x(2).^2;
%plotSingleFunction(f, start, stop);
syms func(x1, x2);
func(x1, x2) = f;
df1 = matlabFunction(diff(func, x1));
df2 = matlabFunction(diff(func, x2));
grad = func_gradient(f, x0);

[x, y, Nsteepest] = steepest_descent(f, x0, target_precision)
[x, y, Nconjugate] = conjugate_gradient(f, x0, target_precision)
%ddf = matlabFunction(diff(func, x, 2));
% Names = ["precision", "bruteforce", "digitiwise", "dichotomy", "golden ratio", "parabolic", "middle point", "chords", "Newton", "Newton-Raphson", "Marquardt"];
% %Names = ["precision", "middle point", "chords", "Newton", "Newton-Raphson", "Marquardt"];
% N = [];
% min_pow = -9;
% max_pow = -1;
% for i = min_pow:max_pow
%     target_precision = 10^(i);
%     [x, y, Nbrute] = bruteforce(f, start, stop, target_precision);
%     [x, y, Ndigit] = digitwise(f, start, stop, target_precision);
%     [x, y, Ndich] = dichotomy(f, start, stop, target_precision);
%     [x, y, Ngold] = goldenratio(f, start, stop, target_precision);
%     [x, y, Nparab] = parabolic(f, start, stop, target_precision);
%     [x, y, Nmid] = middlepoint(f, start, stop, target_precision);
%     [x, y, Nchord] = chord(f, start, stop, target_precision);
%     [x, y, Nnewt] = Newton(f, (start + stop) / 2, target_precision);
%     [x, y, Nnr] = Newton_Raphson(f, (start + stop) / 2, target_precision);
%     [x, y, Nmarq] = Marquardt(f, (start + stop) / 2, target_precision);
% 
%     N = [N;[target_precision, Nbrute, Ndigit, Ndich, Ngold, Nparab, Nmid, Nchord, Nnewt, Nnr, Nmarq]];
%     %N = [N;[target_precision, Nmid, Nchord, Nnewt, Nnr, Nmarq]];
% end
% %plotTableFunctionsLogLog(Names, N, [10^(min_pow), 10^(max_pow)], [0, 10^9])
% plotTableFunctionsLogLog(Names, N, [10^(min_pow), 10^(max_pow)], [1, 80])

N = [];
Names = ["k", "digitwise"];
target_precision = 1e-6;
start_k = 2;
stop_k = 10;
for k = start_k:stop_k
    %Names = [Names, sprintf("k = %d", k)];
    [x, y, Ndigit] = digitwise(f, start, stop, target_precision, k);
    N = [N;[k, Ndigit]];
end
plotTableFunctionsLogLog(Names, N, [start_k, stop_k], [1, 160])

f = @(x) x .* atan(x) - 1/2 * log(1 + x.^2);
plotSingleFunction(f, -4, 4);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
target_precision = 0.001;
%[range_start, range_end] = FindConvergenceInterval(f, 0, @Newton, target_precision);
%[range_start, range_end] = FindConvergenceInterval(f, 0, @Newton_Raphson, target_precision);
%[range_start, range_end] = FindConvergenceInterval(f, 0, @Marquardt, target_precision);

f = @(x) cos(x)./(x.^2);
start = 1;
stop = 12;
plotSingleFunction(f, start, stop);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
L = abs(double(get_lipschitz_const(f, start, stop, 0.000001))) + 1
N1 = [];
for i = 1:7
    target_precision = 10^(-i);
    n = ceil(L * (stop - start) / (target_precision * 2));
    %[x, y, Nbrute] = bruteforce(f, start, stop, target_precision);
    arg = linspace(start, stop, n);
    y = f(arg);
    [fmin, ind] = min(y);
    xmin = arg(ind);
    N1 = [N1 n];
end
N2 = [];
for i = 1:7
    target_precision = 10^(-i);
    [x, y, Npoly] = Polyline(f, start, stop, target_precision)
    N2 = [N2 Npoly];
end
N = [N1;N2]

f = @(x) 1/10 .* x + 2.*sin(4.*x);
start = 0;
stop = 4;
plotSingleFunction(f, start, stop);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
L = abs(double(get_lipschitz_const(f, start, stop, 0.000001))) + 1
N1 = [];
for i = 1:7
    target_precision = 10^(-i);
    n = ceil(L * (stop - start) / (target_precision * 2));
    %[x, y, Nbrute] = bruteforce(f, start, stop, target_precision);
    arg = linspace(start, stop, n);
    y = f(arg);
    [fmin, ind] = min(y);
    xmin = arg(ind);
    N1 = [N1 n];
end
N2 = [];
for i = 1:7
    target_precision = 10^(-i);
    [x, y, Npoly] = Polyline(f, start, stop, target_precision)
    N2 = [N2 Npoly];
end
N = [N1;N2]