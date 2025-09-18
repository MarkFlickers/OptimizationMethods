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
    syms f(x)
    f(x) = func;
    df = diff(f, x);
    xmin = (stop + start) / 2;
    dfmin = df(xmin);
    while abs(dfmin) > target_precision
        if dfmin > 0
            xmin = (start + xmin) / 2;
        else
            xmin = (xmin + stop) / 2;
        end
        dfmin = df(xmin);
    end
    fmin = func(xmin);
end

function [xmin, fmin] = chord(func, start, stop, target_precision)
    syms f(x)
    f(x) = func;
    df = diff(f, x);
    dfa = df(start);
    dfb = df(stop);
    xmin = start - (dfa / (dfa - dfb)) * (start - stop);
    dfmin = df(xmin);
    while abs(dfmin) > target_precision
        if dfmin > 0
            stop = xmin;
            dfb = dfmin;
        else
            start = xmin;
            dfa = dfmin;
        end
        xmin = start - (dfa / (dfa - dfb)) * (start - stop);
        dfmin = df(xmin);
    end
    fmin = func(xmin);
end

function [xmin, fmin] = Newton(func, start, stop, target_precision)
    syms f(x)
    f(x) = func;
    df = diff(f, x);
    ddf = diff(df, x);
    xmin = (stop + start) / 2;
    dfmin = df(xmin);
    ddfmin = ddf(xmin);
    while abs(dfmin) > target_precision
        xmin = xmin - dfmin/ddfmin;
        dfmin = df(xmin);
        ddfmin = ddf(xmin);
    end
    fmin = func(xmin);
end

f = @(x) x.^4 + x.^2 + x.^1 + 1;
[x, y] = bruteforce(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = digitwise(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = dichotomy(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = goldenratio(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = parabolic(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = middlepoint(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = chord(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

[x, y] = Newton(f, -1, 0, 0.000001);

x = num2str(x, 8)
y = num2str(y, 8)

