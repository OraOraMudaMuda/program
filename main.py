import argparse
import os
from imagegen import NoiseVisualizer
from utils import create_mp4_from_pil_images
import torch


def main(song, output_path, seed, hop_length, distance, base_prompt, target_prompts, alpha, guidance_scale, decay_rate,
         boost_factor, boost_threshold):
    # Определяем устройство для выполнения вычислений
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Инициализация визуализатора
    visualizer = NoiseVisualizer(device=device, seed=seed)

    # Загрузка аудиофайла и разбивка его на сегменты
    visualizer.loadSong(song, hop_length=hop_length)

    # Генерацияпредставлений
    latents = visualizer.getSpecCircleLatents(distance=distance)

    # Получение частоты кадров
    fps = visualizer.getFPS()

    # Генерация векторных представлений
    prompt_embeds = visualizer.getPromptEmbeds(basePrompt=base_prompt,
                                               targetPromptChromaScale=target_prompts,
                                               method="slerp",

                                               alpha=alpha,
                                               decay_rate=decay_rate,
                                               boost_factor=boost_factor,
                                               boost_threshold=boost_threshold)

    # Вывод формы
    print(latents.shape)

    # Генерация изображений
    images = visualizer.getVisuals(latents=latents,
                                   promptEmbeds=prompt_embeds,
                                   guidance_scale=guidance_scale)

    # Создание и сохранение видео из сгенерированных изображений и аудиофайла
    create_mp4_from_pil_images(image_array=images,
                               output_path=output_path,
                               song=song,
                               fps=fps)


if __name__ == "__main__":
    # Создание парсера аргументов командной строки
    parser = argparse.ArgumentParser(description="Generate a visualized video based on music and prompts.")

    parser.add_argument("--song", type=str, required=True, help="Path to the song file.")  # Путь к аудиофайлу
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output video.")  # Путь для сохранения видео
    parser.add_argument("--seed", type=int, default=133780085,
                        help="Seed for noise generation.")  # Случайное зерно для генерации шума
    parser.add_argument("--hop_length", type=int, default=377,
                        help="Hop length for audio processing.")  # Длина шага для обработки аудио
    parser.add_argument("--distance", type=float, default=0.3,
                        help="Distance for latent space generation.")  # Расстояние для генерации латентного пространства
    parser.add_argument("--base_prompt", type=str, default="An octopus dancing with cigarettes",
                        help="Base prompt for image generation.")  # Базовая текстовая подсказка для генерации изображений
    parser.add_argument("--target_prompts", type=str, nargs='+', default=[
        "what the dog doin.",
        "giant centipede.",
        "demon scary blood ",
        "man in a suit with juice",
        "massive hamburger yummmm",
        "“broooo thers a beautiful chair",
        "hairy toes hospital",
        "car leaking water flooded",
        "mirror beautiful woman",
        "stop the cap now with a large coffee",
        "turkish rug in a room",
        "horse"],
                        help="List of target prompts for chroma scaling.")  # Список целевых текстовых подсказок для масштабирования хрома
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Alpha value for prompt interpolation.")  # Значение альфа для интерполяции подсказок
    parser.add_argument("--guidance_scale", type=float, default=0,
                        help="Guidance scale for image generation.")  # Масштаб управления для генерации изображений
    parser.add_argument("--decay_rate", type=float, default=0.8)  # Скорость затухания для подсказок
    parser.add_argument("--boost_factor", type=float, default=1.75)  # Коэффициент усиления для подсказок
    parser.add_argument("--boost_threshold", type=float, default=0.4)  # Порог усиления для подсказок

    # Парсинг аргументов командной строки
    args = parser.parse_args()

    # Вызов основной функции
    main(song=args.song,
         output_path=args.output,
         seed=args.seed,
         hop_length=args.hop_length,
         distance=args.distance,
         base_prompt=args.base_prompt,
         target_prompts=args.target_prompts,
         alpha=args.alpha,
         guidance_scale=args.guidance_scale,
         decay_rate=args.decay_rate,
         boost_factor=args.boost_factor,
         boost_threshold=args.boost_threshold)
