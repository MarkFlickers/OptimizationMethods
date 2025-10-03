clc;
clear;

global DERIVATIVE_TYPE_DIFF;
global DERIVATIVE_TYPE_LEFT;
global DERIVATIVE_TYPE_RIGHT;
global DERIVATIVE_TYPE_CENTRAL;
global derivative_type;
global ddf;
global df;

global DERIVATIVE_PRECISION;
global FIRST_DERIVATIVE_COST;
global SECOND_DERIVATIVE_COST;

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

function L = get_lipschitz_const(f, start, stop, precision)
    global df;
    func = @(x) -abs(df(x));
    [xL, L] = bruteforce(func, start, stop, precision);
end

function [xmin, fmin, func_calculations] = bruteforce(f, start, stop, precision)
    n = ceil((stop - start)/precision);
    arg = linspace(start, stop, n);
    y = f(arg);
    func_calculations = n;
    [fmin, ind] = min(y);
    xmin = arg(ind);
end

function [xmin, fmin, func_calculations] = digitwise(f, start, stop, precision)
    func_calculations = 0;
    delta = 1;
    x = start;
    direction = 1;
    yprev = 0;
    while delta > precision
        delta = delta / 4;
        yprev = f(start);
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

function [xmin, fmin, func_calculations] = dichotomy(f, start, stop, target_precision)
    func_calculations = 0;
    d = 0.2*target_precision;
    achieved_precision = (stop - start) / 2;
    while achieved_precision > target_precision
        x1 = (start + stop - d) / 2;
        x2 = (start + stop + d) / 2;
        y1 = f(x1);
        y2 = f(x2);
        func_calculations = func_calculations + 2;
        if y1 <= y2
            stop = x2;
        else
            start = x1;
        end
        achieved_precision = (stop - start) / 2;
    end
    xmin = (stop + start) / 2;
    fmin = f(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations] = goldenratio(f, start, stop, target_precision)
    tau = (sqrt(5) - 1) / 2;
    x1 = start + (1 - tau) * (stop - start);
    x2 = start + tau * (stop - start);
    y1 = f(x1);
    y2 = f(x2);
    func_calculations = 2;
    n = 0;
    achieved_precision = (stop - start) / 2 * tau^n;
    while achieved_precision > target_precision
        if y1 <= y2
            stop = x2;
            x2 = x1;
            x1 = start + stop - x2;
            y2 = y1;
            y1 = f(x1);
            func_calculations = func_calculations + 1;
        else
            start = x1;
            x1 = x2;
            x2 = start + stop - x1;
            y1 = y2;
            y2 = f(x2);
            func_calculations = func_calculations + 1;
        end
        n = n + 1;
        achieved_precision = achieved_precision * tau;
    end
    xmin = (x1 + x2) / 2;
    fmin = f(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations] = parabolic(f, start, stop, target_precision)
    x1 = start;
    x2 = (start + stop) / 2;
    x3 = stop;
    f1 = f(x1);
    f2 = f(x2);
    f3 = f(x3);
    func_calculations = 3;
    while ~((f1 >= f2) && (f2 <= f3))
        if f1 > f3
            x2 = (x2 + x3) / 2;
        else
            x2 = (x1 + x2) / 2;
        end
        f2 = f(x2);
        func_calculations = func_calculations + 1;
    end
    xmin = x2;
    delta = target_precision + 1;

    while delta > target_precision
        a0 = f1;
        a1 = (f2 - f1)/(x2 - x1);
        a2 = 1/(x3 - x2) * (((f3-f1)/(x3-x1))-((f2-f1)/(x2-x1)));
        xminprev = xmin;
        xmin = 1/2*(x1 + x2 - a1/a2);
        fmin = f(xmin);
        func_calculations = func_calculations + 1;

        if xmin < x2
            if fmin >= f2
                x1 = xmin;
                f1 = fmin;
            else
                x1 = x2;
                f1 = f2;
                x2 = xmin;
                f2 = fmin;
            end
        else
            if fmin >= f2
                x3 = xmin;
                f3 = fmin;
            else
                x1 = x2;
                f1 = f2;
                x2 = xmin;
                f2 = fmin;
            end
        end
        delta = abs(xminprev - xmin);
    end
end

function [xmin, fmin, func_calculations] = middlepoint(func, start, stop, target_precision)
    global FIRST_DERIVATIVE_COST;
    func_calculations = 0;
    xmin = (stop + start) / 2;
    dfmin = first_derivative(func, xmin);
    func_calculations = func_calculations + FIRST_DERIVATIVE_COST;
    while abs(dfmin) > target_precision
        if dfmin > 0
            stop = xmin;
        else
            start = xmin;
        end
        xmin = (start + stop) / 2;
        dfmin = first_derivative(func, xmin);
        func_calculations = func_calculations + FIRST_DERIVATIVE_COST;
    end
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations] = chord(func, start, stop, target_precision)
    global FIRST_DERIVATIVE_COST;
    func_calculations = 0;
    dfa = first_derivative(func, start);
    dfb = first_derivative(func, stop);
    xmin = start - (dfa / (dfa - dfb)) * (start - stop);
    dfmin = first_derivative(func, xmin);
    func_calculations = func_calculations + FIRST_DERIVATIVE_COST * 3;
    while abs(dfmin) > target_precision
        if dfmin > 0
            stop = xmin;
            dfb = dfmin;
        else
            start = xmin;
            dfa = dfmin;
        end
        xmin = start - (dfa / (dfa - dfb)) * (start - stop);
        dfmin = first_derivative(func, xmin);
        func_calculations = func_calculations + FIRST_DERIVATIVE_COST;
    end
    fmin = func(xmin);
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

function [xmin, fmin, func_calculations] = Newton_Raphson(func, start, target_precision)
    global FIRST_DERIVATIVE_COST;
    global SECOND_DERIVATIVE_COST;
    func_calculations = 0;
    xmin = start;
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    func_calculations = func_calculations + FIRST_DERIVATIVE_COST + SECOND_DERIVATIVE_COST;
    step = abs(dfmin/ddfmin);
    while abs(dfmin) > target_precision
        xrude = xmin - dfmin/ddfmin;
        dfxrude = first_derivative(func, xrude);
        func_calculations = func_calculations + FIRST_DERIVATIVE_COST;
        tau = dfmin^2 / (dfmin^2 + dfxrude^2);
        xmin = xmin - dfmin/ddfmin * tau;
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

function [xmin, fmin, func_calculations] = Marquardt(func, start, target_precision)
    global FIRST_DERIVATIVE_COST;
    global SECOND_DERIVATIVE_COST;
    func_calculations = 0;
    xmin = start;
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    func_calculations = func_calculations + FIRST_DERIVATIVE_COST + SECOND_DERIVATIVE_COST;
    step = abs(dfmin/ddfmin);
    mu = ddfmin * 10;
    fx = func(xmin);
    func_calculations = func_calculations + 1;
    while abs(dfmin) > target_precision
        xmin = xmin - dfmin/(ddfmin + mu);
        fxnew = func(xmin);
        func_calculations = func_calculations + 1;
        if fx > fxnew
            mu = mu / 2;
        else
            mu = mu * 2;
        end
        fx = fxnew;
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

function [range_start, range_end] = FindConvergenceInterval(f, start, minimizingFunc, target_precision)
    
    prevapprox = start;
    newapprox = -100;
    while(abs(prevapprox - newapprox) > target_precision)
        [x, y] = minimizingFunc(f, newapprox, target_precision);
        if abs(x) < 1e-2
            prevapprox = newapprox;
            newapprox = newapprox * 2;
        else
            newapprox = (newapprox + prevapprox) / 2;
        end
    end
    range_start = prevapprox;

    prevapprox = start;
    newapprox = 100;
    while(abs(prevapprox - newapprox) > target_precision)
        [x, y] = minimizingFunc(f, newapprox, target_precision);
        if abs(x) < 1e-2
            prevapprox = newapprox;
            newapprox = newapprox * 2;
        else
            newapprox = (newapprox + prevapprox) / 2;
        end
    end
    range_end = prevapprox;
end

function [xmin, fmin, func_calculations] = Polyline(func, start, stop, target_precision)
    L = abs(double(get_lipschitz_const(func, start, stop, target_precision)));
    f_a = func(start);
    f_b = func(stop);
    func_calculations = 2;
    x = (f_a - f_b + L*(start + stop)) / (2*L);
    y = (f_a + f_b + L*(start - stop)) / 2;
    x_p_arr = [[x, y]];
    delta_y = target_precision + 1;
    while delta_y > target_precision
        [p, i] = min(x_p_arr(:,2));
        x = x_p_arr(i,1);
        y = func(x);
        func_calculations = func_calculations + 1;
        delta_x = (y - p) / (2*L);
        delta_y = delta_x * (2*L);
        x1 = x - delta_x;
        x2 = x + delta_x;
        p = (y + p) / 2;
        x_p_arr = [x_p_arr(1:i - 1,:); [x1, p]; [x2, p]; x_p_arr(i + 1:end,:)];
    end
    xmin = x;
    fmin = y;
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
    xscale log;
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
start = -1;
stop = 0;
f = @(x) x.^4 + x.^2 + x.^1 + 1;
plotSingleFunction(f, start, stop);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
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

f = @(x) x .* atan(x) - 1/2 * log(1 + x.^2);
plotSingleFunction(f, -4, 4);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
target_precision = 0.001;
[range_start, range_end] = FindConvergenceInterval(f, 0, @Newton, target_precision);
[range_start, range_end] = FindConvergenceInterval(f, 0, @Newton_Raphson, target_precision);
[range_start, range_end] = FindConvergenceInterval(f, 0, @Marquardt, target_precision);

f = @(x) cos(x)./(x.^2);
start = 1;
stop = 12;
plotSingleFunction(f, start, stop);
syms func(x);
func(x) = f;
df = matlabFunction(diff(func, x));
ddf = matlabFunction(diff(func, x, 2));
L = abs(double(get_lipschitz_const(f, start, stop, 0.000001)))
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
    [x, y, Npoly] = Polyline(f, start, stop, target_precision);
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
L = abs(double(get_lipschitz_const(f, start, stop, 0.000001)))
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
    [x, y, Npoly] = Polyline(f, start, stop, target_precision);
    N2 = [N2 Npoly];
end
N = [N1;N2]