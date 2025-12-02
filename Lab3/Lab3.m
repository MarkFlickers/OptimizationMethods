clear
close all

%% ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
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

function [min_arg, min_val, calculations] = annealing(arg, f_vals, initial_point, alpha, T0, nmax)
    single_dimension_len = sqrt(length(arg));
    min_index = 1;
    max_index = single_dimension_len;
    
    calculations = 0;
    get_composite_index = @(i, j) i + ((j - 1) * single_dimension_len);
    
    % Параметры, подобранные эмпирически для сетки ~40x40
    %T0 = 2.0;           % Высокая начальная температура для исследования
    %alpha = 0.99;       % Очень медленное охлаждение
    base_step = max(1, floor(max_index / 8));  % Шаг ~5 для сетки 40x40
    
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
        
        % Ранняя остановка при нахождении глобального минимума
        % (предполагаем, что глобальный минимум известен или можем оценить)
        %if iter > 50 && best_val < -0.01  % Примерное условие
        %    break;
        %end
    end
    
    min_arg = arg(get_composite_index(best_idx(1), best_idx(2)), :);
    min_val = best_val;
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
    end
    
    % Нахождение лучшей особи
    [min_val, best_idx] = min(fitness);
    best_individual = population(best_idx, :);
    min_arg = arg(get_composite_index(best_individual(1), best_individual(2)), :);
end

%% ==================== ФУНКЦИЯ ДЛЯ ЗАПУСКА МНОГОКРАТНЫХ ТЕСТОВ ====================
function run_multiple_tests(func_name, arg, f_vals, global_min_val, global_min_arg, argument_precision, N, algorithm_params)
    % N - количество запусков каждого алгоритма
    % algorithm_params - структура с параметрами для каждого алгоритма
    
    single_dimension_len = sqrt(length(arg));
    
    % Подготовка данных для статистики
    algorithms = {'Pattern Search', 'Random Search', 'Simulated Annealing', 'Genetic Algorithm'};
    n_algorithms = length(algorithms);
    
    % Массивы для хранения результатов
    success_counts = zeros(n_algorithms, N);
    success_rates = zeros(n_algorithms, N);
    avg_calculations = zeros(n_algorithms, 1);
    
    % Для детерминированных алгоритмов (Pattern Search) будем использовать случайные начальные точки
    rng('shuffle'); % Инициализация генератора случайных чисел
    
    fprintf('\n=== Тестирование на функции: %s ===\n', func_name);
    fprintf('Количество запусков каждого алгоритма: %d\n', N);
    fprintf('Глобальный минимум: %.6f\n', global_min_val);
    
    % Запуск каждого алгоритма N раз
    for alg_idx = 1:n_algorithms
        fprintf('\nАлгоритм: %s\n', algorithms{alg_idx});
        
        total_calculations = 0;
        
        for run_idx = 1:N
            % Выбор алгоритма
            switch alg_idx
                case 1 % Pattern Search
                    % Случайная начальная точка
                    initial_point = [randi([1, single_dimension_len]), randi([1, single_dimension_len])];
                    [min_arg, min_val, calc] = pattern_search(arg, f_vals, initial_point, ...
                        algorithm_params.pattern.initial_step, ...
                        algorithm_params.pattern.reduction_factor, ...
                        algorithm_params.pattern.nmax, ...
                        algorithm_params.pattern.target_precision);
                    
                case 2 % Random Search
                    [min_arg, min_val, calc] = random_search(arg, f_vals, algorithm_params.random.nmax);
                    
                case 3 % Simulated Annealing
                    % Случайная начальная точка
                    initial_point = [randi([1, single_dimension_len]), randi([1, single_dimension_len])];
                    [min_arg, min_val, calc] = annealing(arg, f_vals, initial_point, ...
                        algorithm_params.annealing.alpha, algorithm_params.annealing.T0, algorithm_params.annealing.nmax);
                    
                case 4 % Genetic Algorithm
                    [min_arg, min_val, calc] = genetic(arg, f_vals, ...
                        algorithm_params.genetic.population_size, ...
                        algorithm_params.genetic.num_generations, ...
                        algorithm_params.genetic.crossover_rate, ...
                        algorithm_params.genetic.mutation_rate, ...
                        algorithm_params.genetic.selection_method);
            end
            
            total_calculations = total_calculations + calc;
            
            % Проверка, найден ли глобальный минимум
            %if abs(min_val - global_min_val) == 0
            %    success_counts(alg_idx, run_idx) = 1;
            %end

            if norm(global_min_arg - min_arg) <= argument_precision
               success_counts(alg_idx, run_idx) = 1;
            end

            % Вычисление текущей доли успешных запусков
            if run_idx == 1
                success_rates(alg_idx, run_idx) = success_counts(alg_idx, run_idx);
            else
                success_rates(alg_idx, run_idx) = sum(success_counts(alg_idx, 1:run_idx)) / run_idx;
            end
            
            % Прогресс
            if mod(run_idx, max(1, floor(N/10))) == 0
                fprintf('  Запуск %d/%d: успешность = %.2f%%\n', ...
                    run_idx, N, 100 * success_rates(alg_idx, run_idx));
            end
        end
        
        avg_calculations(alg_idx) = total_calculations / N;
        
        fprintf('Итоговая успешность: %.2f%%\n', 100 * success_rates(alg_idx, end));
        fprintf('Среднее число вычислений функции: %.1f\n', avg_calculations(alg_idx));
    end
    
    % Построение графика
    figure('Position', [100, 100, 1200, 600]);
    
    % График 1: Доля успешных запусков
    subplot(1, 2, 1);
    hold on;
    colors = lines(n_algorithms);
    for alg_idx = 1:n_algorithms
        plot(1:N, success_rates(alg_idx, :), 'LineWidth', 2, 'Color', colors(alg_idx, :), ...
            'DisplayName', sprintf('%s (%.1f%%)', algorithms{alg_idx}, 100*success_rates(alg_idx, end)));
    end
    hold off;
    
    xlabel('Количество запусков, N', 'FontSize', 12, 'FontWeight', 'bold');

    ylabel(sprintf('Доля правильных ответов c допуском = %d', argument_precision), 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Сходимость алгоритмов (%s)', func_name), 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    xlim([1, N]);
    ylim([0, 1.05]);
    
    % График 2: Среднее число вычислений функции
    subplot(1, 2, 2);
    bar_data = [avg_calculations, 100*success_rates(:, end)];
    
    % Нормализация для двух осей
    yyaxis left;
    bar(1:n_algorithms, bar_data(:, 1));
    ylabel('Среднее число вычислений функции', 'FontSize', 12, 'FontWeight', 'bold');
    
    yyaxis right;
    plot(1:n_algorithms, bar_data(:, 2), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    ylabel('Итоговая успешность, %', 'FontSize', 12, 'FontWeight', 'bold');
    ylim([0, 105]);
    
    set(gca, 'XTick', 1:n_algorithms, 'XTickLabel', algorithms, 'FontSize', 10);
    title('Эффективность алгоритмов', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Общий заголовок
    sgtitle(sprintf('Сравнение алгоритмов оптимизации (N=%d запусков)', N), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Сохранение результатов в файл
    results_table = table(algorithms', success_rates(:, end), avg_calculations, ...
        'VariableNames', {'Algorithm', 'SuccessRate', 'AvgCalculations'});
    disp(results_table);
end

%% ==================== ОСНОВНОЙ КОД ====================
% Параметры тестирования
N_runs = 100; % Количество запусков каждого алгоритма

%% Функция 1
fprintf('=== АНАЛИЗ ФУНКЦИИ 1 ===\n');
[x1, x2, f] = parse_data_file("Функция_П2.txt");
f_vals_to_minimize = -f;

% Визуализация функции 1
x1_unique = unique(x1);
x2_unique = flip(unique(x2));
Z = reshape(f, length(x1_unique), length(x2_unique))';

figure('Name', 'Функция 1');
contour(x1_unique, x2_unique, Z, 60, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5]);
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Функция 1: Исходные данные', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
axis equal;

% Находим глобальный минимум (максимум исходной функции) для сравнения
[global_max_arg, global_max_val, ~] = bruteforce_discrete([x1 x2], f);
global_min_val = -global_max_val;
global_min_arg = global_max_arg;
fprintf('Глобальный минимум f_min = %.6f в точке (%.6f, %.6f)\n', ...
    global_min_val, global_max_arg(1), global_max_arg(2));

% Параметры алгоритмов для функции 1
algorithm_params_func1 = struct();
algorithm_params_func1.pattern = struct('initial_step', 40, 'reduction_factor', 0.5, 'nmax', 800, 'target_precision', 0.0001);
algorithm_params_func1.random = struct('nmax', 1000);
algorithm_params_func1.annealing = struct('nmax', 500, 'alpha', 0.99, 'T0', 2.0);
algorithm_params_func1.genetic = struct('population_size', 17, 'num_generations', 25, 'crossover_rate', 0.7, 'mutation_rate', 0.3, 'selection_method', 'tournament');

% Запуск многократного тестирования для функции 1
argument_precision = 0;
run_multiple_tests('Function1', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func1);
argument_precision = 1;
run_multiple_tests('Function1', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func1);

%% Функция 2
fprintf('\n\n=== АНАЛИЗ ФУНКЦИИ 2 ===\n');
[x1, x2, f] = parse_data_file("Функция_П4_В2.txt");
f_vals_to_minimize = -f;

% Визуализация функции 2
x1_unique = unique(x1);
x2_unique = flip(unique(x2));
Z = reshape(f, length(x1_unique), length(x2_unique))';

figure('Name', 'Функция 2');
contour(x1_unique, x2_unique, Z, 60, 'LineWidth', 0.5, 'LineColor', [0.5 0.5 0.5]);
xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
title('Функция 2: Исходные данные', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
axis equal;

% Находим глобальный минимум для функции 2
[global_max_arg, global_max_val, ~] = bruteforce_discrete([x1 x2], f);
global_min_val = -global_max_val;
global_min_arg = global_max_arg;
fprintf('Глобальный минимум f_min = %.6f в точке (%.6f, %.6f)\n', ...
    global_min_val, global_max_arg(1), global_max_arg(2));

% Параметры алгоритмов для функции 2 (другие параметры)
algorithm_params_func2 = struct();
algorithm_params_func2.pattern = struct('initial_step', 30, 'reduction_factor', 0.85, 'nmax', 1000, 'target_precision', 0.00001);
algorithm_params_func2.random = struct('nmax', 1000);
algorithm_params_func2.annealing = struct('nmax', 1000, 'alpha', 0.99, 'T0', 2.0);
algorithm_params_func2.genetic = struct('population_size', 20, 'num_generations', 50, 'crossover_rate', 0.85, 'mutation_rate', 0.6, 'selection_method', 'tournament');

% Запуск многократного тестирования для функции 2
argument_precision = 0;
run_multiple_tests('Function2', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func2);
argument_precision = 1;
run_multiple_tests('Function2', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func2);