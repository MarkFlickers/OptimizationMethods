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