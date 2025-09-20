global DERIVATIVE_TYPE_DIFF;
global DERIVATIVE_TYPE_LEFT;
global DERIVATIVE_TYPE_RIGHT;
global DERIVATIVE_TYPE_CENTRAL;
global derivative_type;
global ddf;
global df;

global DERIVATIVE_PRECISION;

DERIVATIVE_TYPE_DIFF = 0;
DERIVATIVE_TYPE_LEFT = DERIVATIVE_TYPE_DIFF + 1;
DERIVATIVE_TYPE_RIGHT = DERIVATIVE_TYPE_LEFT + 1;
DERIVATIVE_TYPE_CENTRAL = DERIVATIVE_TYPE_RIGHT + 1;
derivative_type = DERIVATIVE_TYPE_DIFF;

function [derivative_val] = calculate_left_derivative(f, x, h)
    derivative_val = (f(x) - f(x - h)) / h;
end

function [derivative_val] = calculate_right_derivative(f, x, h)
    derivative_val = (f(x + h) - f(x)) / h;
end

function [derivative_val] = calculate_central_derivative(f, x, h)
    derivative_val = (f(x + h) - f(x - h)) / (h * 2);
end

function [derivative_val] = calculate_second_left_derivative(f, x, h)
    %derivative_val = (f(x) - (2*f(x - h)) - f(x - 2*h)) / h^2;
    derivative_val = (f(x + h) - (2*f(x)) + f(x - h)) / h^2;
end

function [derivative_val] = calculate_second_right_derivative(f, x, h)
    %derivative_val = (f(x + 2*h) - f(x)) / h^2;
    derivative_val = (f(x + h) - (2*f(x)) + f(x - h)) / h^2;
end

function [derivative_val] = calculate_second_central_derivative(f, x, h)
    derivative_val = (f(x + h) - (2*f(x)) + f(x - h)) / h^2;
end

function [derivative_val] = first_derivative(func, arg)
    global DERIVATIVE_TYPE_DIFF;
    global DERIVATIVE_TYPE_LEFT;
    global DERIVATIVE_TYPE_RIGHT;
    global DERIVATIVE_TYPE_CENTRAL;
    global derivative_type;
    global DERIVATIVE_PRECISION;
    global ddf;
    global df;
    switch derivative_type
        case DERIVATIVE_TYPE_DIFF
            %syms f(x);
            %f(x) = func;
            %df = diff(f, x);
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
    global df;
    switch derivative_type
        case DERIVATIVE_TYPE_DIFF
            %syms f(x);
            %f(x) = func;
            %ddf = diff(f, x, 2);
            derivative_val = double(ddf(arg));
        case DERIVATIVE_TYPE_LEFT
            derivative_val = calculate_second_left_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_RIGHT
            derivative_val = calculate_second_right_derivative(func, arg, DERIVATIVE_PRECISION);
        case DERIVATIVE_TYPE_CENTRAL
            derivative_val = calculate_second_central_derivative(func, arg, DERIVATIVE_PRECISION);
    end
end

function [xmin, fmin] = bruteforce(f, start, stop, precision)

    n = ceil((stop - start)/precision);
    x = linspace(start, stop, n);
    y = f(x);
    [fmin, ind] = min(y);
    xmin = x(ind);
end

function [xmin, fmin] = digitwise(f, start, stop, precision)
    delta = 1;
    x = start;
    direction = 1;
    yprev = 0;
    while delta > precision
        delta = delta / 4;
        yprev = f(start);
        x = x + delta*direction;
        y = f(x);
        while y < yprev
            yprev = y;
            xprev = x;
            x = x + delta*direction;
            y = f(x);
        end
        direction = direction * -1;
    end
    fmin = yprev;
    xmin = xprev;
end

function [xmin, fmin] = dichotomy(f, start, stop, target_precision)
    d = 0.2*target_precision;
    achieved_precision = (stop - start) / 2;
    while achieved_precision > target_precision
        x1 = (start + stop - d) / 2;
        x2 = (start + stop + d) / 2;
        y1 = f(x1);
        y2 = f(x2);
        if y1 <= y2
            stop = x2;
        else
            start = x1;
        end
        achieved_precision = (stop - start) / 2;
    end
    xmin = (stop + start) / 2;
    fmin = f(xmin);
end

function [xmin, fmin] = goldenratio(f, start, stop, target_precision)
    tau = (sqrt(5) - 1) / 2;
    x1 = start + (1 - tau) * (stop - start);
    x2 = start + tau * (stop - start);
    y1 = f(x1);
    y2 = f(x2);
    n = 0;
    achieved_precision = (stop - start) / 2 * tau^n;
    while achieved_precision > target_precision
        if y1 <= y2
            stop = x2;
            x2 = x1;
            x1 = start + stop - x2;
            y2 = y1;
            y1 = f(x1);
        else
            start = x1;
            x1 = x2;
            x2 = start + stop - x1;
            y1 = y2;
            y2 = f(x2);
        end
        n = n + 1;
        achieved_precision = achieved_precision * tau;
    end
    xmin = (x1 + x2) / 2;
    fmin = f(xmin);
end

function [xmin, fmin] = parabolic(f, start, stop, target_precision)
    x1 = start;
    x2 = (start + stop) / 2;
    x3 = stop;
    f1 = f(x1);
    f2 = f(x2);
    f3 = f(x3);
    while ~((f1 >= f2) && (f2 <= f3))
        if f1 > f3
            x2 = (x2 + x3) / 2;
        else
            x2 = (x1 + x2) / 2;
        end
        f2 = f(x2);
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

function [xmin, fmin] = middlepoint(func, start, stop, target_precision)
    %syms f(x)
    %f(x) = func;
    %df = diff(f, x);
    xmin = (stop + start) / 2;
    %dfmin = df(xmin);
    dfmin = first_derivative(func, xmin);
    while abs(dfmin) > target_precision
        if dfmin > 0
            stop = xmin;
        else
            start = xmin;
        end
        xmin = (start + stop) / 2;
        dfmin = first_derivative(func, xmin);
        %dfmin = df(xmin);
    end
    fmin = func(xmin);
end

function [xmin, fmin] = chord(func, start, stop, target_precision)
    %syms f(x)
    %f(x) = func;
    %df = diff(f, x);
    %dfa = df(start);
    %dfb = df(stop);
    dfa = first_derivative(func, start);
    dfb = first_derivative(func, stop);
    xmin = start - (dfa / (dfa - dfb)) * (start - stop);
    %dfmin = df(xmin);
    dfmin = first_derivative(func, xmin);
    while abs(dfmin) > target_precision
        if dfmin > 0
            stop = xmin;
            dfb = dfmin;
        else
            start = xmin;
            dfa = dfmin;
        end
        xmin = start - (dfa / (dfa - dfb)) * (start - stop);
        %dfmin = df(xmin);
        dfmin = first_derivative(func, xmin);
    end
    fmin = func(xmin);
end

function [xmin, fmin] = Newton(func, start, target_precision)
    %syms f(x)
    %f(x) = func;
    %df = diff(f, x);
    %ddf = diff(df, x);
    xmin = start;
    %dfmin = df(xmin);
    %ddfmin = ddf(xmin);
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    step = abs(dfmin/ddfmin);
    while abs(dfmin) > target_precision
        xmin = xmin - dfmin/ddfmin;
        prevstep = step;
        step = abs(dfmin/ddfmin);
        %dfmin = df(xmin);
        %ddfmin = ddf(xmin);
        if (step - prevstep) > 0
            xmin = Inf;
            break
        end 
        dfmin = first_derivative(func, xmin);
        ddfmin = second_derivative(func, xmin);
    end
    fmin = func(xmin);
end

function [xmin, fmin] = Newton_Raphson(func, start, target_precision)
    xmin = start;
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    step = abs(dfmin/ddfmin);
    while abs(dfmin) > target_precision
        xrude = xmin - dfmin/ddfmin;
        dfxrude = first_derivative(func, xrude);
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
    end
    fmin = func(xmin);
end

function [xmin, fmin] = Marquardt(func, start, target_precision)
    xmin = start;
    dfmin = first_derivative(func, xmin);
    ddfmin = second_derivative(func, xmin);
    step = abs(dfmin/ddfmin);
    mu = ddfmin * 10;
    fx = func(xmin);
    while abs(dfmin) > target_precision
        xmin = xmin - dfmin/(ddfmin + mu);
        if fx > func(xmin)
            mu = mu / 2;
        else
            mu = mu * 2;
        end
        fx = func(xmin);
        prevstep = step;
        step = abs(dfmin/ddfmin);
        if (step - prevstep) > 0
            xmin = Inf;
            break
        end 
        dfmin = first_derivative(func, xmin);
        ddfmin = second_derivative(func, xmin);
    end
    fmin = func(xmin);
end

target_precision = 0.000001;
DERIVATIVE_PRECISION = target_precision;
start = -1;
stop = 0;
f = @(x) x.^4 + x.^2 + x.^1 + 1;
syms func(x);
func(x) = f;
df = diff(func, x);
ddf = diff(func, x, 2);

xline = linspace(start, stop, 1000);
plot(xline, f(xline))

[x, y] = bruteforce(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = digitwise(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = dichotomy(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = goldenratio(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = parabolic(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = middlepoint(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = chord(f, start, stop, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = Newton(f, (start + stop) / 2, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)


[x, y] = Newton_Raphson(f, (start + stop) / 2, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = Marquardt(f, (start + stop) / 2, target_precision);

x = num2str(x, 8)
y = num2str(y, 8)


f = @(x) x .* atan(x) - 1/2 * log(1 + x.^2);

start_flag = 0;
step = 0.01;
for start = -2:step:2
    [x, y] = Newton(f, start, target_precision);
    if abs(x) < 1e-2
        if start_flag == 0
            range_start = start;
            start_flag = 1;
        end
    elseif start_flag == 1
        range_end = start - step;
        start_flag = 0;
        break
    end
end

range_start, range_end
x = num2str(x, 8)
y = num2str(y, 8)