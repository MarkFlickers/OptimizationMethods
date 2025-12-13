function [min_arg, min_val, calculations] = annealing(arg, f_vals, initial_point, alpha, T0, nmax)
    single_dimension_len = sqrt(length(arg));
    min_index = 1;
    max_index = single_dimension_len;
    
    calculations = 0;
    get_composite_index = @(i, j) i + ((j - 1) * single_dimension_len);
    
    base_step = max(1, floor(max_index / 8));
    
    % Текущее состояние
    current_idx = initial_point;
    current_idx(1) = max(min_index, min(max_index, current_idx(1)));
    current_idx(2) = max(min_index, min(max_index, current_idx(2)));
    
    current_val = f_vals(get_composite_index(current_idx(1), current_idx(2)));
    calculations = calculations + 1;
    
    best_idx = current_idx;
    best_val = current_val;
    
    T = T0;
    
    % Главный цикл с двумя фазами
    for iter = 1:nmax
        % Определяем фазу
        if iter < nmax/2
            % Фаза 1: исследование (большие шаги)
            step_size = base_step * (1 + 2*rand());  % Случайный шаг
            T_phase = T;  % Высокая температура
        else
            % Фаза 2: уточнение (маленькие шаги)
            step_size = max(1, base_step * rand());  % Меньший шаг
            T_phase = T * 0.3;  % Низкая температура для точности
        end
        
        % Генерация новой точки
        angle = 2 * pi * rand();
        dx = round(step_size * cos(angle));
        dy = round(step_size * sin(angle));
        
        new_idx = current_idx + [dx, dy];
        new_idx(1) = max(min_index, min(max_index, new_idx(1)));
        new_idx(2) = max(min_index, min(max_index, new_idx(2)));
        
        % Вычисление
        new_val = f_vals(get_composite_index(new_idx(1), new_idx(2)));
        calculations = calculations + 1;
        
        % Принятие решения с температурой фазы
        delta = new_val - current_val;
        
        if delta < 0
            accept = true;
        else
            probability = exp(-delta / (T_phase + eps));
            accept = (rand() < probability);
        end
        
        if accept
            current_idx = new_idx;
            current_val = new_val;
        end
        
        % Обновление лучшего
        if new_val < best_val
            best_idx = new_idx;
            best_val = new_val;
        end
        
        % Охлаждение
        T = T * alpha;
        
    end
    
    min_arg = arg(get_composite_index(best_idx(1), best_idx(2)), :);
    min_val = best_val;
end