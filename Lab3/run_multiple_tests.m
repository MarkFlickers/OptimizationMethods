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