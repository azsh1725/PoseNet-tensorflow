# PoseNet (tensorflow)

Сеть для предсказания координат по фото, реализованная по следующей [статье](https://arxiv.org/abs/1505.07427).

### Подготовка:

- Установить необходимые библиотеки из `requirements.txt`, если предполагается тренировка на GPU, то нужно вместо 
`tensorflow` поставить `tensorflow-gpu`.
- Скачать предобученные веса для GoogLeNet [отсюда](https://www.dropbox.com/sh/axnbpd1oe92aoyd/AADpmuFIJTtxS7zkL_LZrROLa?dl=0),
файл называется `googlenet.npy`, создать директорию `/model/pretrained_models` и положить туда скачанный файл.

- Распаковать архив с данными в `/data`, так, чтобы существовала следующая директория 
`/data/camera_relocalization_sample_dataset`

### Обучение:

Для запуска обучения необходимо запустить файл `train.py`.
Пример команды запуска - ```python train.py```


### Инференс:
Для запуска модели на инференс необходимо запустить файл `test.py` и передать первым аргументом путь до картинки для 
которой хочется получить предсказание.
Пример команды запуска - ```python test.py ./data/camera_relocalization_sample_dataset/images/img_0_0_1542098919031247900.png```