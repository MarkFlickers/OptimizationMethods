%clc;
%clear;

global GRADIENT_TYPE_DIFF;
global GRADIENT_TYPE_FORWARD;
global GRADIENT_TYPE_BACKWARD;
global GRADIENT_TYPE_CENTRAL;
global gradient_type;
global df1;
global df2;

global GRADIENT_PRECISION;
global ONE_D_MINIMIZATION_PRECISION;
global GRADIENT_COST;
global HESSIAN_COST;

GRADIENT_TYPE_DIFF      = 0;
GRADIENT_TYPE_FORWARD   = 1;
GRADIENT_TYPE_BACKWARD  = 2;
GRADIENT_TYPE_CENTRAL   = 3;

gradient_type = GRADIENT_TYPE_DIFF;

if gradient_type == GRADIENT_TYPE_DIFF
    GRADIENT_COST = 2;
    HESSIAN_COST = 4;
else
    GRADIENT_COST = 4;
    HESSIAN_COST = 3;
end

GRADIENT_COST = 2;

function grad = calculate_central_gradient(f, x, h)
    n = length(x);
    for i = 1:n
        x_forward = x;
        x_forward(i) = x_forward(i) + h;
        x_backward = x;
        x_backward(i) = x_backward(i) - h;
        grad(i) = (f(x_forward) - f(x_backward)) / (2 * h);
    end
end

function grad = calculate_forward_gradient(f, x, h)
    n = length(x);
    for i = 1:n
        x_perturbed = x;
        x_perturbed(i) = x_perturbed(i) + h;
        grad(i) = (f(x_perturbed) - f0) / h;
    end
end

function grad = calculate_backward_gradient(f, x, h)
    n = length(x);
    for i = 1:n
        x_perturbed = x;
        x_perturbed(i) = x_perturbed(i) - h;
        grad(i) = (f0 - f(x_perturbed)) / h;
    end
end


function [grad] = func_gradient(f, x)
    global GRADIENT_TYPE_DIFF;
    global GRADIENT_TYPE_CENTRAL;
    global GRADIENT_TYPE_FORWARD;
    global GRADIENT_TYPE_BACKWARD;
    global gradient_type;
    global GRADIENT_PRECISION;
    global df1;
    global df2;
    f = @(x) f(x(1), x(2));
    switch gradient_type
        case GRADIENT_TYPE_DIFF
            grad = [double(df1(x(1), x(2))), double(df2(x(1), x(2)))];
        case GRADIENT_TYPE_CENTRAL
            grad = calculate_central_gradient(f, x, GRADIENT_PRECISION);
        case GRADIENT_TYPE_FORWARD
            grad = calculate_forward_gradient(f, x, GRADIENT_PRECISION);
        case GRADIENT_TYPE_BACKWARD
            grad = calculate_backward_gradient(f, x, GRADIENT_PRECISION);
    end
end

function [H, eval_count] = numerical_hessian(f, x, h)
% Оптимизированная версия с подсчетом числа вычислений функции
%
% Входные параметры:
%   f - функция многих переменных (function handle)
%   x - точка, в которой вычисляется Гессиан (вектор)
%   h - шаг для конечных разностей (опционально)
%
% Выходные параметры:
%   H - матрица Гессе (n x n)
%   eval_count - количество вычислений функции

    if nargin < 3
        h = 1e-6;
    end
    
    x = x(:);
    n = length(x);
    H = zeros(n, n);
    eval_count = 0;
    
    % Предварительно вычисляем значения в точках x ± h*e_i
    f_plus = zeros(n, 1);
    f_minus = zeros(n, 1);
    
    for i = 1:n
        x_plus = x;
        x_plus(i) = x_plus(i) + h;
        f_plus(i) = f(x_plus);
        
        x_minus = x;
        x_minus(i) = x_minus(i) - h;
        f_minus(i) = f(x_minus);
        
        eval_count = eval_count + 2;
    end
    
    f0 = f(x);
    eval_count = eval_count + 1;
    
    % Заполняем диагональные элементы
    for i = 1:n
        H(i,i) = (f_plus(i) - 2*f0 + f_minus(i)) / (h^2);
    end
    
    % Заполняем внедиагональные элементы
    for i = 1:n
        for j = i+1:n
            x_plus_plus = x;
            x_plus_plus(i) = x_plus_plus(i) + h;
            x_plus_plus(j) = x_plus_plus(j) + h;
            f_plus_plus = f(x_plus_plus);
            
            x_plus_minus = x;
            x_plus_minus(i) = x_plus_minus(i) + h;
            x_plus_minus(j) = x_plus_minus(j) - h;
            f_plus_minus = f(x_plus_minus);
            
            x_minus_plus = x;
            x_minus_plus(i) = x_minus_plus(i) - h;
            x_minus_plus(j) = x_minus_plus(j) + h;
            f_minus_plus = f(x_minus_plus);
            
            x_minus_minus = x;
            x_minus_minus(i) = x_minus_minus(i) - h;
            x_minus_minus(j) = x_minus_minus(j) - h;
            f_minus_minus = f(x_minus_minus);
            
            eval_count = eval_count + 4;
            
            H(i,j) = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * h^2);
            H(j,i) = H(i,j); % Симметрия
        end
    end
end

function [H, evaluations_count] = calculate_Hessian(f, x)
    global GRADIENT_TYPE_DIFF;
    global GRADIENT_TYPE_CENTRAL;
    global GRADIENT_TYPE_FORWARD;
    global GRADIENT_TYPE_BACKWARD;
    global gradient_type;
    global GRADIENT_PRECISION;

    if gradient_type == GRADIENT_TYPE_DIFF
        syms sym_func(x1, x2);
        sym_func(x1, x2) = f;
        Hessian = hessian(sym_func);
        H = double(Hessian(x(1), x(2)));
        evaluations_count = 4;
    else
        f = @(x) f(x(1), x(2));
        [H, evaluations_count] = numerical_hessian(f, x, GRADIENT_PRECISION);
    end
end
    

function [xmin, fmin, func_calculations] = digitwise(f, start, precision, k)
    global ONE_D_MINIMIZATION_PRECISION
    precision = ONE_D_MINIMIZATION_PRECISION;
    delta = 0.01;
    x = start;
    direction = 1;
    y = f(x);
    func_calculations = 1;
    while delta > precision
        delta = delta / k;
        yprev = y;
        xprev = x;
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

function [xmin, fmin, func_calculations, trajectory] = steepest_descent(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    global GRADIENT_COST;
    gradient_calculations = 0;
    % Метод наискорейшего спуска
    trajectory = [x0];
    x = x0;
    func_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    
    for iter = 1:max_iter
        grad = func_gradient(f,x);
        %func_calculations = func_calculations + 1;
        %func_calculations = func_calculations + GRADIENT_COST;
        gradient_calculations = gradient_calculations + 1;
        % Критерий остановки
        if ~isempty(p.Results.MinPoint)
            min_point = p.Results.MinPoint;
            if norm(min_point - x) < epsilon
                break;
            end
        elseif norm(grad) < epsilon
            break;
        end
      
        % Поиск оптимального шага
        [alpha, Fmin, calcs] = digitwise(@(a) func(x - a * grad), 0, epsilon/10, 4);
        func_calculations = func_calculations + calcs;
        
        x_new = x - alpha * grad;
        x = x_new;
        trajectory = [trajectory; x];
    end
    
    xmin = x;
    fmin = func(x);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = conjugate_gradient(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    global GRADIENT_COST;
    % Метод сопряженных градиентов
    trajectory = [x0];
    x = x0;
    func_calculations = 0;
    gradient_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    n = length(x0);
    
    grad = func_gradient(f,x);
    %func_calculations = func_calculations + GRADIENT_COST;
    gradient_calculations = gradient_calculations + 1;
    d = -grad;

    iters = 0;
    
    for iter = 1:max_iter
        % Поиск оптимального шага
        [alpha, Fmin, calcs] = digitwise(@(a) func(x + a * d), 0, epsilon/10, 4);
        func_calculations = func_calculations + calcs;

        x = x + alpha * d;
        trajectory = [trajectory; x];
        grad_prev = grad;
        grad = func_gradient(f,x);
        % func_calculations = func_calculations + GRADIENT_COST;
        gradient_calculations = gradient_calculations + 1;

        beta = norm(grad)^2 / norm(grad_prev)^2;
        iters = iters + 1;
        if iters == 3 * n
            beta = 0;
            iters = 0;
        end
        d = -grad + beta * d;

        if ~isempty(p.Results.MinPoint)
            min_point = p.Results.MinPoint;
            if norm(min_point - x) < epsilon
                break;
            end
        elseif norm(grad) < epsilon
            break;
        end
    end
    xmin = x;
    fmin = func(x);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = Newton(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    global GRADIENT_COST;
    gradient_calculations = 0;
    trajectory = [x0];
    x = x0;
    func_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2)); 
    grad = func_gradient(f,x);
    %func_calculations = func_calculations + GRADIENT_COST;
    gradient_calculations = gradient_calculations + 1;

    [A, hessian_cost] = calculate_Hessian(f, x0);
    func_calculations = func_calculations + hessian_cost;
    Ainv = inv(A);
    for iter = 1:max_iter
        % Поиск оптимального шага
        x = (x' - Ainv * grad')';
        trajectory = [trajectory; x];
        grad = func_gradient(f,x);
        %func_calculations = func_calculations + GRADIENT_COST;
        gradient_calculations = gradient_calculations + 1;
        if ~isempty(p.Results.MinPoint)
            min_point = p.Results.MinPoint;
            if norm(min_point - x) < epsilon
                break;
            end
        elseif norm(grad) < epsilon
            break;
        end
        A = calculate_Hessian(f, x);
        func_calculations = func_calculations + hessian_cost;
        Ainv = inv(A);
    end
    xmin = x;
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = Right_Simplex(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    trajectory = [x0];
    x = x0;
    func_calculations = 1;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    reduction_coeff = 1/2;
    edge_len = 1;
    f_prev = func(x);
    xk1 = x;
    xk2 = [x(1) + ((sqrt(2 + 1) - 1)/(2 * sqrt(2)))*edge_len, x(2) + ((sqrt(2 + 1) + 2 - 1)/(2 * sqrt(2)))*edge_len];
    xk3 = [x(1) + ((sqrt(2 + 1) + 2 - 1)/(2 * sqrt(2)))*edge_len, x(2) + ((sqrt(2 + 1) - 1)/(2 * sqrt(2)))*edge_len];
   
    for iter = 1:max_iter
        simplex = [xk1; xk2; xk3];
        fs = [func(simplex(1,:)); func(simplex(2,:)); func(simplex(3,:))];
        [vals, index] = sort(fs);
        xk1 = simplex(index(1),:);
        xk2 = simplex(index(2),:);
        xk3 = simplex(index(3),:);
        func_calculations = func_calculations + 2;
        x_mirrored = 2/2*(xk1 + xk2) - xk3;
        
        f_min = func(x_mirrored);
        func_calculations = func_calculations + 1;
        if f_min >= vals(3)
            % if min in xk1
            edge_len = edge_len * reduction_coeff;
            %x = xk1;
            xk2 = xk1 + reduction_coeff * (xk2 - xk1);
            xk3 = xk1 + reduction_coeff * (xk3 - xk1);
            f_prev = vals(1);
            trajectory = [trajectory; xk1];
        else
            % if min in x_mirrored
            xk3 = x_mirrored;
            f_prev = f_min;
            trajectory = [trajectory; x_mirrored];
        end
        
        if ~isempty(p.Results.MinPoint)
            min_point = p.Results.MinPoint;
            if norm(min_point - x_mirrored) < epsilon
                break;
            end
        elseif edge_len < epsilon
            break;
        end
    end
    xmin = x;
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = Cyclic_Coordinate(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    trajectory = [x0];
    n = length(x0);
    x = x0;
    func_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    basis = eye(n);
    xprev = x0;
    fprev = func(xprev);

    j = 1;
    for iter = 1:max_iter
        ej = basis(j,:);
        [alpha, Fmin, calcs] = digitwise(@(a) func(x + a * ej), 0, epsilon/10, 4);
        func_calculations = func_calculations + calcs;
        %func_calculations = func_calculations + 1;
        fcur = Fmin;
        x = x + alpha * ej;
        trajectory = [trajectory; x];
        j = j + 1;
        if j == n + 1
            j = 1;
            if ~isempty(p.Results.MinPoint)
                min_point = p.Results.MinPoint;
                if norm(min_point - x) < epsilon
                    break;
                end
            elseif norm(x - xprev) < epsilon
                break;
            end
            fprev = fcur;
            xprev = x;
        end
    end
    xmin = x;
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = Hooke_Jeeves(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    trajectory = [x0];
    n = length(x0);
    x = x0;
    deltas = [1 1];
    gamma = 4;
    scale_coeff = 0.5;
    func_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    basis = eye(n);
    xprev = x0;
    j = 1;
    for iter = 1:max_iter
        %func_calculations = func_calculations + 1;
        ej = basis(j,:);
        y1 = x - deltas(j) * ej;
        y2 = x + deltas(j) * ej;
        if(func(x) > func(y1))
            x = y1;
            func_calculations = func_calculations + 1;
        elseif(func(x) > func(y2))
            x = y2;
            func_calculations = func_calculations + 2;
        else
            func_calculations = func_calculations + 2;
        end
        j = j + 1;
        if j == n + 1
            j = 1;
            if(x == xprev)
                if ~isempty(p.Results.MinPoint)
                    min_point = p.Results.MinPoint;
                    if norm(min_point - x) < epsilon
                        break;
                    else
                        deltas = deltas ./ gamma;
                    end
                elseif norm(deltas) < epsilon
                    break;
                else
                    deltas = deltas ./ gamma;
                end
            end
            xprev = x;
            x = x - scale_coeff*(x - xprev);
            trajectory = [trajectory; x];
        end
    end
    xmin = x;
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function [xmin, fmin, func_calculations, trajectory] = Random_Search(f, x0, epsilon, varargin)
    p = inputParser;
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    parse(p, varargin{:});
    trajectory = [x0];
    n = length(x0);
    x = x0;
    alpha = 1;
    gamma = 4;
    m = 3*n;
    func_calculations = 0;
    max_iter = 1000000;
    func = @(x) f(x(1), x(2));
    j = 1;
    ksi = randi([-1 1],1,2);
    for iter = 1:max_iter
        y = x + alpha * ksi / norm(ksi);
        if(func(x) > func(y))
            x = y;
            func_calculations = func_calculations + 1;
            trajectory = [trajectory; x];
            continue
        end
        j = j + 1;
        if j == m + 1
            j = 1;
            if ~isempty(p.Results.MinPoint)
                min_point = p.Results.MinPoint;
                if norm(min_point - x) < epsilon
                    break;
                else
                    alpha = alpha / gamma;
                end
            elseif alpha < epsilon
                break;
            else
                alpha = alpha / gamma;
            end
        end
        ksi = randi([-10 10],1,2);
    end
    xmin = x;
    fmin = func(xmin);
    func_calculations = func_calculations + 1;
end

function contour_plot_with_optimization(f, x_limits, y_limits, level_step, varargin)
% Построение линий уровня функции двух переменных с точкой минимума и траекторией
%
% Входные параметры:
%   f - функция двух переменных (function handle) f = @(x1,x2) ...
%   x_limits - пределы по x [x_min, x_max]
%   y_limits - пределы по y [y_min, y_max]  
%   level_step - шаг между линиями уровня
%
% Опциональные параметры (пары "ключ-значение"):
%   'MinPoint' - точка минимума [x_min, y_min]
%   'Trajectory' - массив точек траектории Nx2
%   'Title' - заголовок графика
%   'ColorMap' - цветовая схема ('parula', 'jet', 'hot', etc.)
%   'ShowColorbar' - показывать цветовую шкалу (true/false)

    % Парсинг опциональных параметров
    p = inputParser;
    addRequired(p, 'f', @(x) isa(x, 'function_handle'));
    addRequired(p, 'x_limits', @(x) isnumeric(x) && length(x)==2);
    addRequired(p, 'y_limits', @(x) isnumeric(x) && length(x)==2);
    addRequired(p, 'level_step', @isnumeric);
    addParameter(p, 'MinPoint', [], @(x) isnumeric(x) && (isempty(x) || length(x)==2));
    addParameter(p, 'Trajectory', [], @(x) isnumeric(x) && (isempty(x) || size(x,2)==2));
    addParameter(p, 'Title', "Линии уровня функции", @isstring);
    addParameter(p, 'ColorMap', 'false', @ischar);
    addParameter(p, 'ShowColorbar', true, @islogical);
    
    parse(p, f, x_limits, y_limits, level_step, varargin{:});
    
    % Создание сетки
    x1 = linspace(x_limits(1), x_limits(2), 300);
    x2 = linspace(y_limits(1), y_limits(2), 300);
    [X1, X2] = meshgrid(x1, x2);
    
    % Вычисление значений функции на сетке
    Z = arrayfun(@(a,b) f(a,b), X1, X2);
    
    % Определение уровней для линий уровня
    z_min = min(Z(:));
    z_max = max(Z(:));
    %levels = z_min:level_step:z_max;
    levels = [25, 20, 15, 10, 5, 2, 1, 0.5, 0.25, 0.175, 0.1, 0.05, 0.01, 0.003];
    
    % Создание графика
    figure;
    hold on;
    
    % Построение линий уровня
    %contourf(X1, X2, Z, levels, 'LineWidth', 0.5);
    %contour(X1, X2, Z, levels, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5]);
    contour(X1, X2, Z, 40, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5], 'DisplayName', 'Линии уровня функции');
    
    % Настройка цветовой схемы
    %colormap(p.Results.ColorMap);
    if p.Results.ShowColorbar
    %    colorbar;
    end
    
    % Отображение точки минимума (если задана)
    if ~isempty(p.Results.MinPoint)
        min_point = p.Results.MinPoint;
        plot(min_point(1), min_point(2), 'ro', 'MarkerSize', 10, ...
             'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black', ...
             'LineWidth', 2, 'DisplayName', 'Точка минимума');
        
        % Подпись точки минимума
        %text(min_point(1), min_point(2), '  Настоящий минимум', ...
        %     'Color', 'red', 'FontWeight', 'bold', 'FontSize', 10);
    end
    
    % Отображение траектории (если задана)
    if ~isempty(p.Results.Trajectory)
        trajectory = p.Results.Trajectory;
        
        % Рисуем траекторию
        plot(trajectory(:,1), trajectory(:,2), 'b-', 'LineWidth', 2, ...
             'DisplayName', 'Траектория поиска');
        
        % Рисуем точки траектории
        plot(trajectory(:,1), trajectory(:,2), 'bo', 'MarkerSize', 4, ...
             'MarkerFaceColor', 'white');
        
        % Выделяем начальную точку
        if size(trajectory,1) > 0
            plot(trajectory(1,1), trajectory(1,2), 'gs', 'MarkerSize', 8, ...
                 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'black', ...
                 'LineWidth', 2, 'DisplayName', 'Начальная точка');
        end
        
        % Выделяем конечную точку  
        if size(trajectory,1) > 1
            plot(trajectory(end,1), trajectory(end,2), 'bd', 'MarkerSize', 8, ...
                 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black', ...
                 'LineWidth', 2, 'DisplayName', 'Конечная точка');
        end
    end
    
    % Настройка графика
    xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
    title(p.Results.Title, 'FontSize', 14, 'FontWeight', 'bold');
    
    grid on;
    axis equal;
    axis([x_limits, y_limits]);
    
    % Добавление легенды, если есть что показывать
    if ~isempty(p.Results.MinPoint) || ~isempty(p.Results.Trajectory)
        legend('show', 'Location', 'best');
    end
    
    hold off;
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

% target_precision =1e-5;
% GRADIENT_PRECISION = target_precision/10;
% ONE_D_MINIMIZATION_PRECISION = target_precision/10;
% a = 1000;
% f = @(x1, x2) x1.^2 + a * x2.^2;
% x0 = [1, -1];
% syms func(x1, x2);
% func(x1, x2) = f;
% df1 = matlabFunction(diff(func, x1));
% df2 = matlabFunction(diff(func, x2));
% fu = @(x) f(x(1), x(2));
% min = [0 0];
% grad = func_gradient(f, x0);
% xrange = [-2, 2];
% yrange = [-2, 2];
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Title="Тестовая функция", MinPoint=min);
% [x, y, Nsteepest, trajectory] = steepest_descent(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Наискорейший спуск");
% [x, y, Nconjugatem, trajectory] = conjugate_gradient(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Сопряжённый градиент");
% [x, y, NNewton, trajectory] = Newton(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Ньютона");
% [x, y, NSimplex, trajectory] = Right_Simplex(f, x0, target_precision, MinPoint=min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Правильный симплекс");
% [x, y, NCoordinate, trajectory] = Cyclic_Coordinate(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Циклический покоординатный спуск");
% [x, y, NHookeJeeves, trajectory] = Hooke_Jeeves(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Хука-Дживса");
% [x, y, NRandom, trajectory] = Random_Search(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод случайного поиска");

% target_precision =1e-3;
% GRADIENT_PRECISION = target_precision/10;
% ONE_D_MINIMIZATION_PRECISION = target_precision/10;
% x0 = [42, 55];
% f = @(x1, x2) 129*x1^2 - 256*x1*x2 + 129*x2^2 - 51*x1 - 149*x2 - 27;
% syms func(x1, x2);
% func(x1, x2) = f;
% df1 = matlabFunction(diff(func, x1));
% df2 = matlabFunction(diff(func, x2));
% fu = @(x) f(x(1), x(2));
% min = [50 50];
% grad = func_gradient(f, x0);
% xrange = [40, 60];
% yrange = [40, 60];
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Title="Тестовая функция", MinPoint=min);
% [x, y, Nsteepest, trajectory] = steepest_descent(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Наискорейший спуск");
% [x, y, Nconjugatem, trajectory] = conjugate_gradient(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Сопряжённый градиент");
% [x, y, NNewton, trajectory] = Newton(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Ньютона");
% [x, y, NSimplex, trajectory] = Right_Simplex(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Правильный симплекс");
% [x, y, NCoordinate, trajectory] = Cyclic_Coordinate(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Циклический покоординатный спуск");
% [x, y, NHookeJeeves, trajectory] = Hooke_Jeeves(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Хука-Дживса");
% [x, y, NRandom, trajectory] = Random_Search(f, x0, target_precision);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод случайного поиска");


% target_precision =1e-5;
% GRADIENT_PRECISION = target_precision/10;
% ONE_D_MINIMIZATION_PRECISION = target_precision/10;
% ONE_D_MINIMIZATION_PRECISION = 1e-4;
% x0 = [-1, 1];
% f = @(x1, x2) 100*(x1^2 - x2)^2 + (x1 - 1)^2;
% syms func(x1, x2);
% func(x1, x2) = f;
% df1 = matlabFunction(diff(func, x1));
% df2 = matlabFunction(diff(func, x2));
% grad = func_gradient(f, x0);
% min = [1 1];
% xrange = [-2, 2];
% yrange = [-2, 2];
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Title="Тестовая функция", MinPoint=min);
% [x, y, Nsteepest, trajectory] = steepest_descent(f, x0, target_precision);
% diffSteepest = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Наискорейший спуск");
% [x, y, Nconjugatem, trajectory] = conjugate_gradient(f, x0, target_precision);
% diffConjugate = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Сопряжённый градиент");
% [x, y, NNewton, trajectory] = Newton(f, x0, target_precision);
% diffNewton = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Ньютона");
% [x, y, NSimplex, trajectory] = Right_Simplex(f, x0, target_precision);
% diffSimplex = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Правильный симплекс");
% [x, y, NCoordinate, trajectory] = Cyclic_Coordinate(f, x0, target_precision);
% diffCoordinate = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Циклический покоординатный спуск");
% [x, y, NHookeJeeves, trajectory] = Hooke_Jeeves(f, x0, target_precision);
% diffHookeJeeves = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Хука-Дживса");
% [x, y, NRandom, trajectory] = Random_Search(f, x0, target_precision);
% diffRandom = norm(x-min);
% contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод случайного поиска");

target_precision =1e-3;
DERIVATIVE_PRECISION = target_precision/10;
ONE_D_MINIMIZATION_PRECISION = target_precision/10;
x0 = [0, 0];
%x0 = [-5 0];
f = @(x1, x2) (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2;
syms func(x1, x2);
func(x1, x2) = f;
df1 = matlabFunction(diff(func, x1));
df2 = matlabFunction(diff(func, x2));
grad = func_gradient(f, x0);
min = fminsearch(@(x) f(x(1), x(2)), [50, 50]);
xrange = [-5, 5];
yrange = [-5, 5];
contour_plot_with_optimization(f, xrange, yrange, 0.1, Title="Тестовая функция", MinPoint=min);
[x, y, Nsteepest, trajectory] = steepest_descent(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Наискорейший спуск");
[x, y, Nconjugatem, trajectory] = conjugate_gradient(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Сопряжённый градиент");
[x, y, NNewton, trajectory] = Newton(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Ньютона");
[x, y, NSimplex, trajectory] = Right_Simplex(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Правильный симплекс");
[x, y, NCoordinate, trajectory] = Cyclic_Coordinate(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Циклический покоординатный спуск");
[x, y, NHookeJeeves, trajectory] = Hooke_Jeeves(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод Хука-Дживса");
[x, y, NRandom, trajectory] = Random_Search(f, x0, target_precision);
contour_plot_with_optimization(f, xrange, yrange, 0.1, Trajectory=trajectory, Title="Метод случайного поиска");