function [x1, x2, f] = parse_data_file(filename)
% Чтение данных с использованием textscan (более эффективно для больших файлов)

    % Проверяем существование файла
    if ~isfile(filename)
        error('Файл "%s" не найден.', filename);
    end
    
    % Открываем файл
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Не удалось открыть файл "%s".', filename);
    end
    
    % Читаем данные с помощью textscan
    % Формат: число, число, число (с экспоненциальной нотацией)
    data = textscan(fileID, '%f %f %f');
    fclose(fileID);
    
    % Извлекаем колонки
    x1 = data{1};
    x2 = data{2};
    f = data{3};
    
    % Удаляем строки с NaN (если есть некорректные данные)
    valid_indices = ~isnan(x1) & ~isnan(x2) & ~isnan(f);
    x1 = x1(valid_indices);
    x2 = x2(valid_indices);
    f = f(valid_indices);
    
    fprintf('Успешно прочитано %d строк из файла "%s"\n', length(x1), filename);
end

function [max_arg, max_val, calculations] = bruteforce_discrete(arg, f_vals)
    max_arg = arg(1);
    max_val = f_vals(1);
    calculations = 1;
    for i = 2:length(f_vals)
        calculations = calculations + 1;
        if(f_vals(i) > max_val)
            max_val = f_vals(i);
            max_arg = arg(i,:);
        end
    end
end

function [min_arg, min_val, calculations] = annealing(arg, f_vals, c0, T0, L, nmax)
    single_dimension_len = sqrt(length(arg));
    n = 0;
    c = c0 / (c0 + 1);
    min_index = 1;
    calculations = 1;
    arg1_index_n = min_index;
    arg2_index_n = min_index;
    
    %f_rnd = @(x, T) x + L * (rand(1) - 0.5) * (T0/T);
    %f_rnd = @(x, T) x + L * (2*rand(1) - 1) * max(0.1, T/T0) * (single_dimension_len/10);
     f_rnd = @(x, T) x + L * (rand(1) - 0.5) * (T / T0);
    get_composite_index = @(index_1, index_2) index_1 + ((index_2 - 1) * single_dimension_len);
    
    x_n = arg(get_composite_index(arg1_index_n, arg2_index_n), :);
    x_best = x_n;
    f_best = f_vals(get_composite_index(arg1_index_n, arg2_index_n));
    while(n < nmax)
        %T = T0 * exp(-n/(c0 * nmax));
        T = T0 * exp(-c * n / nmax);
        arg1_index_n1 = round(f_rnd(arg1_index_n, T));
        if(arg1_index_n1 > single_dimension_len)
            arg1_index_n1 = single_dimension_len;
        elseif(arg1_index_n1 < min_index)
            arg1_index_n1 = min_index;
        end
        arg2_index_n1 = round(f_rnd(arg2_index_n, T));
        if(arg2_index_n1 > single_dimension_len)
            arg2_index_n1 = single_dimension_len;
        elseif(arg2_index_n1 < min_index)
            arg2_index_n1 = min_index;
        end
        x_n1 = arg(get_composite_index(arg1_index_n1, arg2_index_n1), :);

        f_n = f_vals(get_composite_index(arg1_index_n, arg2_index_n));
        f_n1 = f_vals(get_composite_index(arg1_index_n1, arg2_index_n1));
        if(f_n > f_n1)
            P = 1;
        else
            P = exp(-(f_n1 - f_n)/T);

        end
        calculations = calculations + 1;
        
        if(rand(1) < P)
            arg1_index_n = arg1_index_n1;
            arg2_index_n = arg2_index_n1;
            x_n = x_n1;
            f_n = f_n1;
        end

        if (f_best > f_n)
            x_best = x_n;
            f_best = f_n;
        end

        n = n + 1;
    end
    min_arg = x_best;
    min_val = f_best;
end

[x1, x2, f] = parse_data_file("Функция_П2.txt");
f_vals_to_minimize = -f;

x1_unique = unique(x1);
x2_unique = flip(unique(x2));
Z = reshape(f, length(x1_unique), length(x2_unique))';


figure;
hold on;

contour(x1_unique, x2_unique, Z, 40, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5], 'DisplayName', 'Линии уровня функции');

[max_arg, max_val, NBruteforce]= bruteforce_discrete([x1 x2], f);
[max_arg, max_val, NAnnealing]= annealing([x1 x2], -f, 10, 100, 50, 400);
max_val = -max_val;

