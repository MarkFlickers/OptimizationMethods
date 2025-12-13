clear
close all


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
algorithm_params_func2.pattern = struct('initial_step', 30, 'reduction_factor', 0.5, 'nmax', 1000, 'target_precision', 0.0001);
algorithm_params_func2.random = struct('nmax', 1000);
algorithm_params_func2.annealing = struct('nmax', 1000, 'alpha', 0.99, 'T0', 1.75);  %0.99/0.995
algorithm_params_func2.genetic = struct('population_size', 30, 'num_generations', 37, 'crossover_rate', 0.9, 'mutation_rate', 0.9, 'selection_method', 'tournament');

% Запуск многократного тестирования для функции 2
argument_precision = 0;
run_multiple_tests('Function2', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func2);
argument_precision = 1;
run_multiple_tests('Function2', [x1 x2], f_vals_to_minimize, global_min_val, global_min_arg, argument_precision, N_runs, algorithm_params_func2);