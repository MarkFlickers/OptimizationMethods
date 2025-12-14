function [min_arg, min_val, calculations] = pattern_search(arg, f_vals, initial_point, initial_step, reduction_factor, nmax, target_precision)
    single_dimension_len = sqrt(length(arg));
    min_index = 1;
    max_index = single_dimension_len;
    base_pattern = [
         0, -1;  % вверх
         1,  0;  % вправо
         0,  1;  % вниз
        -1,  0;  % влево
         1, -1;  % вправо-вверх
         1,  1;  % вправо-вниз
        -1,  1;  % влево-вниз
        -1, -1   % влево-вверх
    ];
   
    current_idx = initial_point;
    step_size = initial_step;
    calculations = 0;

    get_search_pattern = @(step_size) round(base_pattern * step_size);
    get_composite_index = @(index_1, index_2) index_1 + ((index_2 - 1) * single_dimension_len);
    
    % Начальное значение
    current_val = f_vals(get_composite_index(current_idx(1), current_idx(2)));
    calculations = calculations + 8;
    
    best_idx = current_idx;
    best_val = current_val;
    
    for iter = 1:nmax
        
        % Исследующий поиск - проверяем соседние точки
        improved = false;
        
        % Определяем шаблон поиска (4- или 8-связная окрестность)
        pattern = get_search_pattern(step_size);
        
        for p = 1:size(pattern, 1)
            % Новая кандидатная точка
            new_idx = round(current_idx + pattern(p, :));
            
            % Ограничение в пределах сетки
            new_idx(1) = max(min_index, min(max_index, new_idx(1)));
            new_idx(2) = max(min_index, min(max_index, new_idx(2)));
            
            % Пропускаем если точка не изменилась
            if all(new_idx == current_idx)
                continue;
            end
            
            % Вычисляем значение функции
            new_val = f_vals(get_composite_index(new_idx(1), new_idx(2)));
            calculations = calculations + 8;
            
            % Если нашли улучшение
            if new_val < current_val
                current_idx = new_idx;
                current_val = new_val;
                improved = true;
                break; % Переходим к следующей итерации
            end

        end
        
        % Обновляем лучшее решение
        if current_val < best_val
            best_idx = current_idx;
            best_val = current_val;
        end
        
        % Адаптация шага
        if improved
            % Успешный поиск - можно увеличить шаг (опционально)
            step_size = step_size * 1.1;
        else
            % Неудачный поиск - уменьшаем шаг
            step_size = step_size * reduction_factor;
        end
        
        % Критерий остановки
        if step_size < target_precision
            break;
        end

        if calculations > nmax
            break;
        end
    end
    
    % Возвращаем лучшую найденную точку
    min_arg = arg(get_composite_index(best_idx(1), best_idx(2)), :);
    min_val = best_val;
end