function [xmin, ymin] = bruteforce(func, start, stop, precision)

    n = ceil((stop - start)/precision);
    x = linspace(start, stop, n);
    y = func(x);
    [ymin, ind] = min(y);
    xmin = x(ind);
end

function [xmin, ymin] = digitwise(func, start, stop, precision)
    delta = 1;
    x = start;
    direction = 1;
    while delta > precision
        delta = delta / 4;
        yprev = func(start);
        x = x + delta*direction;
        y = func(x);
        while y < yprev
            yprev = y;
            xprev = x;
            x = x + delta*direction;
            y = func(x);
        end
        direction = direction * -1;
    end
    ymin = yprev;
    xmin = xprev;
end

function [xmin, ymin] = dichotomy(func, start, stop, target_precision)
    d = 0.2*target_precision;
    achieved_precision = (stop - start) / 2;
    while achieved_precision > target_precision
        x1 = (start + stop - d) / 2;
        x2 = (start + stop + d) / 2;
        y1 = func(x1);
        y2 = func(x2);
        if y1 <= y2
            stop = x2;
        else
            start = x1;
        end
        achieved_precision = (stop - start) / 2;
    end
    xmin = (stop + start) / 2;
    ymin = func(xmin);
end

function [xmin, ymin] = goldenratio(func, start, stop, target_precision)
    tau = (sqrt(5) - 1) / 2;
    x1 = start + (1 - tau) * (stop - start);
    x2 = start + tau * (stop - start);
    y1 = func(x1);
    y2 = func(x2);
    n = 0;
    achieved_precision = (stop - start) / 2 * tau^n;
    while achieved_precision > target_precision
        if y1 <= y2
            stop = x2;
            x2 = x1;
            x1 = start + stop - x2;
            y2 = y1;
            y1 = func(x1);
        else
            start = x1;
            x1 = x2;
            x2 = start + stop - x1;
            y1 = y2;
            y2 = func(x2);
        end
        n = n + 1;
        achieved_precision = achieved_precision * tau;
    end
    xmin = (x1 + x2) / 2;
    ymin = func(xmin);
end

function [xmin, ymin] = parabolic(func, start, stop, target_precision)
    
    xmin = 0;
    ymin = func(xmin);
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