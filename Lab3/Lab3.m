clear

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
    calculations = calculations + 1;
    
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
            calculations = calculations + 1;
            
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
    end
    
    % Возвращаем лучшую найденную точку
    min_arg = arg(get_composite_index(best_idx(1), best_idx(2)), :);
    min_val = best_val;
end

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

function [min_arg, min_val, calculations] = genetic(arg, f_vals, population_size, num_generations, crossover_rate, mutation_rate, selection_method)
    function population = initialize_population(population_size, min_index, max_index)
        population = [randi([min_index, max_index], population_size, 1), randi([min_index, max_index], population_size, 1)];
    end

    function fitness = evaluate_fitness(population, f_vals, get_composite_index)
        n = size(population, 1);
        fitness = zeros(n, 1);
        for i = 1:n
            idx = get_composite_index(population(i, 1), population(i, 2));
            fitness(i) = f_vals(idx);
        end
    end

    function offspring = crossover(population, fitness, crossover_rate, selection_method)
        population_size = size(population, 1);
        offspring = [];
        
        while size(offspring, 1) < population_size
            % Выбор родителей
            parent1 = select_parent(population, fitness, selection_method);
            parent2 = select_parent(population, fitness, selection_method);
            
            if rand() < crossover_rate
                % Одноточечное скрещивание
                crossover_point = randi(2);
                if crossover_point == 1
                    child1 = [parent1(1), parent2(2)];
                    child2 = [parent2(1), parent1(2)];
                else
                    child1 = parent1;
                    child2 = parent2;
                end
                offspring = [offspring; child1; child2];
            else
                % Без скрещивания - родители переходят в потомки
                offspring = [offspring; parent1; parent2];
            end
        end
        
        % Обрезка до нужного размера
        offspring = offspring(1:population_size, :);
    end

    function parent = select_parent(population, fitness, method)
        switch method
            case 'tournament'
                % Турнирная селекция
                tournament_size = 3;
                tournament_indices = randperm(size(population, 1), tournament_size);
                [~, best_idx] = min(fitness(tournament_indices));
                parent = population(tournament_indices(best_idx), :);
                
            case 'roulette'
                % Селекция рулеткой (для минимизации инвертируем приспособленность)
                max_fit = max(fitness);
                inverted_fitness = max_fit - fitness + eps;
                probabilities = inverted_fitness / sum(inverted_fitness);
                cum_probs = cumsum(probabilities);
                r = rand();
                selected_idx = find(cum_probs >= r, 1);
                parent = population(selected_idx, :);
                
            otherwise
                error('Неизвестный метод селекции: %s', method);
        end
    end

    function mutated_population = mutate(population, mutation_rate, min_index, max_index)
    % Мутация - случайное изменение генов
        mutated_population = population;
        for i = 1:size(population, 1)
            if rand() < mutation_rate
                % Мутация первого гена
                mutated_population(i, 1) = randi([min_index, max_index]);
            end
            if rand() < mutation_rate
                % Мутация второго гена
                mutated_population(i, 2) = randi([min_index, max_index]);
            end
        end
    end

    function [new_population, new_fitness] = selection(population, fitness, offspring, offspring_fitness, method)
    % Селекция - формирование нового поколения
        combined_population = [population; offspring];
        combined_fitness = [fitness; offspring_fitness];
        
        switch method
            case 'tournament'
                % Элитная селекция + турнир
                population_size = size(population, 1);
                
                % Сохраняем лучшую особь (элитизм)
                [~, best_idx] = min(combined_fitness);
                new_population = combined_population(best_idx, :);
                new_fitness = combined_fitness(best_idx);
                
                % Добираем остальных турниром
                remaining_size = population_size - 1;
                for i = 1:remaining_size
                    tournament_indices = randperm(size(combined_population, 1), 3);
                    [~, best_tournament_idx] = min(combined_fitness(tournament_indices));
                    selected_idx = tournament_indices(best_tournament_idx);
                    new_population = [new_population; combined_population(selected_idx, :)];
                    new_fitness = [new_fitness; combined_fitness(selected_idx)];
                end
                
            case 'roulette'
                % Селекция рулеткой
                population_size = size(population, 1);
                max_fit = max(combined_fitness);
                inverted_fitness = max_fit - combined_fitness + eps;
                probabilities = inverted_fitness / sum(inverted_fitness);
                
                selected_indices = roulette_selection(probabilities, population_size);
                new_population = combined_population(selected_indices, :);
                new_fitness = combined_fitness(selected_indices);
        end
    end

    function selected_indices = roulette_selection(probabilities, num_selections)
    % probabilities - вектор вероятностей выбора каждой особи
    % num_selections - сколько особей нужно отобрать
    
        cum_probs = cumsum(probabilities); % Создаем кумулятивную сумму вероятностей
        selected_indices = zeros(num_selections, 1);
        
        for i = 1:num_selections
            r = rand(); % Случайное число от 0 до 1
            selected_indices(i) = find(r <= cum_probs, 1, 'first'); % Находим первый элемент, который превышает r
        end
    end


    single_dimension_len = sqrt(length(arg));
    min_index = 1;
    max_index = single_dimension_len;
    calculations = 0;
    
    get_composite_index = @(index_1, index_2) index_1 + ((index_2 - 1) * single_dimension_len);
    
    % 1. Генерация начальной популяции
    population = initialize_population(population_size, min_index, max_index);
    
    % Оценка начальной приспособленности
    fitness = evaluate_fitness(population, f_vals, get_composite_index);
    calculations = calculations + population_size;
    
    % Главный цикл по поколениям
    for gen = 1:num_generations
        % 2. Скрещивание
        offspring = crossover(population, fitness, crossover_rate, selection_method);
        
        % 3. Мутация
        offspring = mutate(offspring, mutation_rate, min_index, max_index);
        
        % Оценка приспособленности потомков
        offspring_fitness = evaluate_fitness(offspring, f_vals, get_composite_index);
        calculations = calculations + size(offspring, 1);
        
        % 4. Селекция
        [population, fitness] = selection(population, fitness, offspring, offspring_fitness, selection_method);
        
        % Сохранение истории
        [best_fit, best_idx] = min(fitness);
        
        % Проверка условий выхода (ранняя остановка)
        % if gen > 10
        %     break;
        % end
    end
    
    % Нахождение лучшей особи
    [min_val, best_idx] = min(fitness);
    best_individual = population(best_idx, :);
    min_arg = arg(get_composite_index(best_individual(1), best_individual(2)), :);
end

[x1, x2, f] = parse_data_file("Функция_П2.txt");
f_vals_to_minimize = -f;

x1_unique = unique(x1);
x2_unique = flip(unique(x2));
Z = reshape(f, length(x1_unique), length(x2_unique))';


figure;
hold on;

contour(x1_unique, x2_unique, Z, 60, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5], 'DisplayName', 'Линии уровня функции');
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Тестовая функция', 'FontSize', 14, 'FontWeight', 'bold');

grid on;
axis equal;
hold off;

[max_arg, max_val, NBruteforce] = bruteforce_discrete([x1 x2], f);
[min_arg, min_val, NPattern] = pattern_search([x1 x2], -f, [1, 1], 40, 0.5, 800, 0.001);
[min_arg, min_val, NRandom] = random_search([x1 x2], -f, 1000);
[min_arg, min_val, NGenetic] = genetic([x1 x2], -f, 15, 50, 0.6, 0.03, 'roulette');
[min_arg, min_val, NAnnealing]= annealing([x1 x2], -f, 10, 100, 50, 400);
%max_val = -min_val;

[x1, x2, f] = parse_data_file("Функция_П4_В2.txt");
f_vals_to_minimize = -f;

x1_unique = unique(x1);
x2_unique = flip(unique(x2));
Z = reshape(f, length(x1_unique), length(x2_unique))';


figure;
hold on;

contour(x1_unique, x2_unique, Z, 60, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5], 'DisplayName', 'Линии уровня функции');
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Тестовая функция', 'FontSize', 14, 'FontWeight', 'bold');

grid on;
axis equal;
hold off;

[max_arg, max_val, NBruteforce] = bruteforce_discrete([x1 x2], f);
[min_arg, min_val, NPattern] = pattern_search([x1 x2], -f, [1, 1], 40, 0.5, 800, 0.001);
[min_arg, min_val, NRandom] = random_search([x1 x2], -f, 1000);
[min_arg, min_val, NGenetic] = genetic([x1 x2], -f, 10, 40, 0.7, 0.05, 'roulette');
[min_arg, min_val, NAnnealing]= annealing([x1 x2], -f, 10, 100, 50, 400);
