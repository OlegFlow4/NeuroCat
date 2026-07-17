import os
import sys
import time
import subprocess
import ctypes

# ANSI коды для статических цветов
ORANGE = '\033[38;5;214m'
GREY = '\033[90m'
GREEN = '\033[38;5;46m' # Добавили зеленый
RESET = '\033[0m'

# Данные для скачивания
MODEL_REPO = "bartowski/gemma-2-9b-it-GGUF"
MODEL_FILE = "gemma-2-9b-it-Q4_K_M.gguf"
DOWNLOAD_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
DISPLAY_NAME = "Gemma-2-9B-It Q4_K_M"

# Путь, где лежит сам скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

# Заглушка для логов C++, чтобы скрыть спам
def _log_callback(level, message, user_data):
    pass
# Обязательно сохраняем в глобальную переменную, чтобы сборщик мусора Python не удалил callback
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)(_log_callback)

def get_color(percent):
    """Генерирует ANSI RGB код для плавного перехода от красного (0%) к зеленому (100%)"""
    r = int(255 * (1 - percent))
    g = int(255 * percent)
    b = 0
    return f"\033[38;2;{r};{g};{b}m"

def install_dependencies():
    """Проверяет и красиво устанавливает библиотеки, если их нет"""
    packages = ["requests", "llama-cpp-python"]
    missing = []
    
    # Проверяем наличие нужных модулей
    for pkg in packages:
        try:
            if pkg == "llama-cpp-python":
                import llama_cpp
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
            
    if not missing:
        return 

    print(f"{GREY}Настройка нейрокота: загрузка библиотек...{RESET}")
    
    # Запускаем pip install в фоновом режиме
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-q"] + missing,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    percent = 0.0
    bar_length = 30
    
    # Рисуем полоску, пока pip работает в фоне
    for line in process.stdout:
        if percent < 0.95:
            percent += 0.005 
        
        filled_len = int(bar_length * percent)
        bar = '█' * filled_len + '-' * (bar_length - filled_len)
        color = get_color(percent)
        sys.stdout.write(f"\r{GREY}Установка модулей {color}[{bar}] {int(percent * 100)}%{RESET}")
        sys.stdout.flush()
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n{GREY}Ошибка! Похоже, нет компилятора установленного в системе.{RESET}")
        sys.exit(1)
        
    # Завершаем полоску до 100%
    color = get_color(1.0)
    bar = '█' * bar_length
    sys.stdout.write(f"\r{GREY}Установка модулей {color}[{bar}] 100%{RESET}\n")
    sys.stdout.flush()
    time.sleep(1.5)

def download_with_custom_bar():
    """Кастомная функция скачивания с цветной полоской"""
    import requests 
    
    print(f"{GREY}Начинаю загрузку (~5.4 GB). Пожалуйста, подожди...{RESET}")
    try:
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        block_size = 2 * 1024 * 1024 
        downloaded = 0
        bar_length = 30 
        
        with open(MODEL_PATH, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = downloaded / total_size
                    filled_len = int(bar_length * percent)
                    
                    bar = '█' * filled_len + '-' * (bar_length - filled_len)
                    color = get_color(percent)
                    
                    sys.stdout.write(f"\r{GREY}{DISPLAY_NAME} {color}[{bar}] {int(percent * 100)}%{RESET}")
                    sys.stdout.flush()
        
        print(f"\n{GREY}Загрузка успешно завершена!{RESET}")
    except Exception as e:
        print(f"\n{GREY}Произошла ошибка при скачивании: {e}{RESET}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        sys.exit(1)

def main():
    # 1. Сначала скачиваем/устанавливаем библиотеки (если надо)
    install_dependencies()

    # 2. Очищаем терминал
    os.system('cls' if os.name == 'nt' else 'clear')

    # 3. Оранжевое приветствие с текстовым котом
    print(f"{ORANGE}  /\\_/\\  ")
    print(f" ( o.o ) ")
    print(f"  > ^ <  {RESET}")
    print(f"{ORANGE}Привет я нейрокот который сделан на базе Gemma{RESET}")

    # 4. Проверка наличия файла нейросети
    if os.path.exists(MODEL_PATH):
        print(f"{GREY}Модель уже найдена в папке. Пропускаем загрузку.{RESET}")
    else:
        print(f"{GREY}Для общения с нейрокотом нужно загрузить нейросеть из интернета. Загрузить?{RESET}")
        
        answer = input().strip().lower()

        if answer not in ['да', 'da', 'yes']:
            sys.exit(0)

        download_with_custom_bar()

    # Импортируем модули Llama
    from llama_cpp import Llama, llama_log_set

    # Отключаем английские предупреждения и логи C++ на корню
    try:
        llama_log_set(LOG_CALLBACK, ctypes.c_void_p(0))
    except Exception:
        pass

    # 5. Загрузка модели в память
    print(f"{GREY}Пробуждаю нейрокота...{RESET}")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,         
            n_gpu_layers=-1, 
            verbose=False       
        )
    except Exception as e:
        print(f"{GREY}Ошибка при загрузке модели: {e}{RESET}")
        sys.exit(1)

    print(f"{GREY}Нейрокот готов! (Для выхода напиши 'выход'){RESET}")
    print(f"{ORANGE}NeuroCat v1.2 {GREEN}Release{RESET}\n")
    
    # 6. Чат-цикл
    while True:
        user_input = input(f"{ORANGE}> {RESET}").strip()
        
        if user_input.lower() in ['выход', 'exit']:
            break
        
        if not user_input:
            continue

        prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"

        print(GREY, end="")
        
        stream = llm(
            prompt,
            max_tokens=1024,
            stop=["<end_of_turn>", "<start_of_turn>"],
            stream=True
        )

        for chunk in stream:
            text = chunk['choices'][0]['text']
            print(text, end="", flush=True)
        
        print(f"{RESET}\n")

if __name__ == "__main__":
    main()