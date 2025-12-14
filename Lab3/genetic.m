function [min_arg, min_val, calculations] = genetic(arg, f_vals, population_size, num_generations, crossover_rate, mutation_rate, selection_method, nmax)
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
        
        if calculations > nmax
            break;
        end
    end

    % Нахождение лучшей особи
    [min_val, best_idx] = min(fitness);
    best_individual = population(best_idx, :);
    min_arg = arg(get_composite_index(best_individual(1), best_individual(2)), :);
end

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