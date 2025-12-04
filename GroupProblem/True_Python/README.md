# CI Birds

## Описание
Проект по птицам с автоматическим запуском тестов

## Требования
- **Python** версии 3.14 (рекомендуется)

## Установка
1. Убедитесь, что у вас установлен Python 3.14
2. Клонируйте репозиторий или скачайте исходный код проекта

## Использование

### Запуск скрипта
Основной запускаемый файл - `pyrun.py`

#### Обязательные аргументы:
- `--config <путь>` - путь к конфигурационному файлу в формате JSON
- `--data_root <путь>` - путь к директории с входными файлами

#### Опциональные аргументы:
- `--verbose` - вывод подробной информации по анализу правильности ходов

### Пример запуска из командной строки:
```bash
python pyrun.py --config GroupProblem/True_Python/data/test_config.json --data_root GroupProblem/True_Python/data
```

```bash
python pyrun.py --config GroupProblem/True_Python/data/test_config.json --data_root GroupProblem/True_Python/data --verbose
```

## Настройка окружения разработки

### Запуск в Visual Studio Code
Для удобного запуска в Visual Studio Code можно создать конфигурацию запуска:

```json
{
    "name": "My Python launch",
    "type": "debugpy",
    "request": "launch",
    "program": "GroupProblem/True_Python/pyrun.py",
    "python": "C:/Users/User/AppData/Local/Programs/Python/Python314/python.exe",
    "args": [
        "--config", 
        "GroupProblem/True_Python/data/test_config.json", 
        "--data_root", 
        "GroupProblem/True_Python/data",
        "--verbose"
    ],
    "console": "integratedTerminal"
}
```

### Примечания:
1. Убедитесь, что путь к интерпретатору Python соответствует вашей системе


## Структура проекта
```
.
├── GroupProblem/
│   ├── True_Python/
│   |   ├── data/                           # Директория с данными и конфигурацией
│   |   │   ├── test_config.json            # Конфиг с перечислением наборов входных данных
│   |   │   ├── test_config_template.json   # Пример конфига с перечислением наборов входных данных
│   |   │   ├── inputs                      # Директория с файлами входных данных
|   |   |   └── outputs                     # Директория с подпапками с результатами работы программы
|   |   |
│   |   ├── src/                            # Директория с исходниками
│   |   │   ├── __init__.py                 # Интерфейсная часть (?)
│   |   │   ├── astar.py                    # Искатель пути
│   |   │   ├── BranchIntegrity.py          # Парсер входных данных и валидатор найденного пути
|   |   |   └── temp.py                     # Для логов
|   |   |
│   |   ├── tests/                          # Директория с данными и конфигурацией
│   |   │   ├── __init__.py                 # Интерфейсная часть
|   |   |   └── e2e.py                      # Ядро тестового окружения
|   |   |
│   |   ├── pyrun.py                        # Основной исполняемый файл
|   |   └── README.md                       # Этот файл
|   .
.
```

## Конфигурационный файл
Конфигурационный файл в формате JSON определяет, какие имена входных файлов искать в директории `data_root`. Конфигурационный файл можно написать по образцу из `test_config_template.json`.

## Поддержка
Для получения спама в коносльный вывод используйте `--verbose` при запуске.