function [min_arg, min_val, calculations] = random_search(arg, f_vals, nmax)
    single_dimension_len = sqrt(length(arg));
    min_index = 1;
    max_index = single_dimension_len;
    
    calculations = 0;
    
    % Лучшее решение
    best_idx = [min_index, min_index];
    get_composite_index = @(index_1, index_2) index_1 + ((index_2 - 1) * single_dimension_len);
    
    best_val = f_vals(get_composite_index(best_idx(1), best_idx(2)));
    calculations = calculations + 1;
    
    for i = 1:nmax
        % Генерация случайной точки
        idx1 = randi([min_index, max_index]);
        idx2 = randi([min_index, max_index]);
        
        % Вычисление значения функции
        current_val = f_vals(get_composite_index(idx1, idx2));
        calculations = calculations + 1;
        
        % Обновление лучшего решения
        if current_val < best_val
            best_val = current_val;
            best_idx = [idx1, idx2];
        end
    end
    
    min_arg = arg(get_composite_index(best_idx(1), best_idx(2)), :);
    min_val = best_val;
end