function [xmin, ymin] = bruteforce(func, start, stop, precision)

    n = ceil((stop - start)/precision);
    x = linspace(start, stop, n);
    y = func(x);
    [ymin, ind] = min(y);
    xmin = x(ind);
end

f = @(x) x.^4 + x.^2 + x.^1 + 1;
[x, y] = bruteforce(f, -1, 0, 0.000001);

s = num2str(x, 8)
s = num2str(y, 8)

