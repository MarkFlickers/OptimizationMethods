function run_multiple_tests(func_name, arg, f_vals, global_min_val, global_min_arg, argument_precision, N, algorithm_params)
    % N - максимальное значение nmax
    % algorithm_params - структура с параметрами для каждого алгоритма
    
    single_dimension_len = sqrt(length(arg));
    
    % Подготовка данных для статистики
    algorithms   = {'Pattern Search', 'Random Search', 'Simulated Annealing', 'Genetic Algorithm'};
    n_algorithms = numel(algorithms);
    
    % ---------------- НАСТРОЙКА СЕТКИ ПО nmax ----------------
    step  = 10;                 % при необходимости замените на нужный шаг
    nmax_values = 10:step:N;   % именно здесь nmax = 10:step:N
    K = numel(nmax_values);    % количество разных значений nmax
    
    % Массивы для хранения результатов
    success_rates   = zeros(n_algorithms, K);  % процент успеха для каждого nmax
    avg_calculations = zeros(n_algorithms, K); % среднее число вычислений для каждого nmax
    
    rng('shuffle'); % Инициализация генератора случайных чисел
    
    fprintf('\n=== Тестирование на функции: %s ===\n', func_name);
    fprintf('Число прогонов на каждое значение nmax: 100\n');
    fprintf('Диапазон nmax: [%d : %d : %d]\n', 10, step, N);
    fprintf('Глобальный минимум: %.6f\n', global_min_val);
    
    for alg_idx = 1:n_algorithms
        fprintf('\nАлгоритм: %s\n', algorithms{alg_idx});
        
        for k = 1:K
            nmax = nmax_values(k);
            
            success_count   = 0;  % число успешных попыток при данном nmax
            total_calc_for_nmax = 0;
            
            % ------------ 100 запусков для фиксированного nmax ------------
            for run_idx = 1:100
                switch alg_idx
                    case 1 % Pattern Search
                        initial_point = [randi([1, single_dimension_len]), randi([1, single_dimension_len])];
                        [min_arg, min_val, calc] = pattern_search( ...
                            arg, f_vals, initial_point, ...
                            algorithm_params.pattern.initial_step, ...
                            algorithm_params.pattern.reduction_factor, ...
                            nmax, ...
                            algorithm_params.pattern.target_precision);
                        
                    case 2 % Random Search
                        [min_arg, min_val, calc] = random_search(arg, f_vals, nmax);
                        
                    case 3 % Simulated Annealing
                        initial_point = [randi([1, single_dimension_len]), randi([1, single_dimension_len])];
                        [min_arg, min_val, calc] = annealing( ...
                            arg, f_vals, initial_point, ...
                            algorithm_params.annealing.alpha, ...
                            algorithm_params.annealing.T0, ...
                            nmax);
                        
                    case 4 % Genetic Algorithm
                        [min_arg, min_val, calc] = genetic( ...
                            arg, f_vals, ...
                            algorithm_params.genetic.population_size, ...
                            algorithm_params.genetic.num_generations, ...
                            algorithm_params.genetic.crossover_rate, ...
                            algorithm_params.genetic.mutation_rate, ...
                            algorithm_params.genetic.selection_method, ...
                            nmax);
                end
                
                total_calc_for_nmax = total_calc_for_nmax + calc;
                
                % Условие успешности по аргументу
                if norm(global_min_arg - min_arg) <= argument_precision
                    success_count = success_count + 1;
                end
            end
            
            % Доля успешных запусков (из 100) для данного nmax
            success_rates(alg_idx, k)   = success_count / 100;
            avg_calculations(alg_idx, k) = total_calc_for_nmax / 100;
            
            fprintf('nmax = %4d: успех = %.2f%%, среднее вычислений = %.1f\n', ...
                nmax, 100*success_rates(alg_idx, k), avg_calculations(alg_idx, k));
        end
    end
    
    % ---------------- ПОСТРОЕНИЕ ГРАФИКОВ ----------------
    figure('Position', [100, 100, 1200, 600]);
    hold on;
    colors = lines(n_algorithms);
    
    % График: доля успешных запусков в зависимости от nmax
    for alg_idx = 1:n_algorithms
        plot(nmax_values, success_rates(alg_idx, :), ...
            'LineWidth', 2, ...
            'Color', colors(alg_idx, :), ...
            'DisplayName', sprintf('%s (%.1f%% при nmax=%d)', ...
                algorithms{alg_idx}, 100*success_rates(alg_idx, end), nmax_values(end)));
    end
    hold off;
    
    xlabel('n_{max}', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(sprintf('Доля правильных ответов c допуском = %g', argument_precision), ...
           'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Сходимость алгоритмов (%s)', func_name), ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    xlim([nmax_values(1), nmax_values(end)]);
    ylim([0, 1.05]);
    
    sgtitle(sprintf('Сравнение алгоритмов оптимизации (100 запусков на каждое n_{max})'), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Сохранение таблицы итоговой успешности при максимальном nmax
    final_success = success_rates(:, end);
    final_avg_calc = avg_calculations(:, end);
    
    results_table = table(algorithms', final_success, final_avg_calc, ...
        'VariableNames', {'Algorithm', 'SuccessRate_at_max_nmax', 'AvgCalculations_at_max_nmax'});
    disp(results_table);
end
