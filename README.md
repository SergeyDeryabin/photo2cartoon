<div align='center'>
  <img src='./images/title.png'>
</div>

# Фото -> Мультяшный рисунок(Photo to Cartoon)

Русская версия | [中文版](README_IE.md) | English Version (README_EN.md)

[Minivision](https://www.minivision.cn/)': Проект перевода фото в мультяшный рисунок (Фото -> Мультяшный рисунок( photo-to-cartoon translation project is Photo to Cartoon)) является проектом с открытыми исходниками в этом репозитарии( repo) и вы можете попробовать нашу WeChat-мини-программу  "AI Cartoon Show", через сканирование QR-кода, расположенного внизу ниже.   try our .

<div>
  <img src='./images/QRcode.jpg' height='150px' width='150px'>
</div>

Вы можете также попробовать эту страницу: [https://ai.minivision.cn/#/coreability/cartoon](https://ai.minivision.cn/#/coreability/cartoon)

**Обновления**
- `2021.12.2`:  Выполните эту модель на хосте тиражированиия Replicate [Run this model on Replicate](https://beta.replicate.ai/hao-qiang/photo2cartoon).
- `2020.12.2`: Выпуен релиз [photo2cartoon-paddle](https://github.com/minivision-ai/photo2cartoon-paddle).
- `2020.12.1`: Добавили тестовую onnx-модель, подробности смотрите [test_onnx.py](./test_onnx.py).

## Введение

Цель мультипликационной стилизации портрета(portrait cartoon stylization) состоит в преобразовании реальных фотографий в мультяшные рисунки с деталями ID-информации и текстуры портрета. Мы используем метод сети Generative Adversarial Network(Порождающая Соперничающая Сеть)  чтобы понять отображение фото в мультяшный рисунок. Рассматривая трудность в получении парных данных и несоответствующей фигуры ввода и вывода( paired data and the non-corresponding shape of input and output), мы принимаем стиль перевода непарного  изображения.

Результаты метода CycleGAN, классического метода перевода непарного изображения(unpaired image translation method), часто имеют очевидные артефакты и нестабильны. Недавно, Ким и др. предложили  новую функцию нормализации( normalization function; AdaLIN) и модуль внимания(attention module) в документе "U-GAT-IT" и достигают изящных результатов в программе переноса селфи-фото в  аниме-рисунок selfie2anime.

Отличаясь от преувеличенного стиля аниме, наш мультяшный стиль более реалистичен и содержит определенную ID-информацию. С этой целью мы добавляем Потерю ID Лица (Face ID Loss; расстояние косинуса  ID-признака между входным изображением и мультяшным изображением; cosine distance of ID features between input image and cartoon image), чтобы достигнуть инвариантности идентификационных данных(identity invariance). 

Мы предлагаем  метод нормализации мягкого адаптивного экземпляра-слоя(Soft Adaptive Layer-Instance Normalization; Soft-AdaLIN) , который вместе сплавляет статистику кодированных признаков и декодирования признаков в де-стандартизации. 

Чтобы улучшить производительность в прогрессивно отображать ход процесса, на основе U-GAT-IT, перед кодером и после декодера представлены два модуля песочных часов.

Мы также предварительно обрабатываем данные к фиксированному шаблону(образцу;  fixed pattern), чтобы уменьшить трудность оптимизации. Дополнительную информацию смотрите ниже.

<div align='center'>
  <img src='./images/results.png'>
</div>

## Начало

### Требования
- python 3.6
- pytorch 1.4
- tensorflow-gpu 1.14
- face-alignment
- dlib
- onnxruntime

### Клонировать

```
git clone https://github.com/minivision-ai/photo2cartoon.git
cd ./photo2cartoon
```

### Загрузить

Google -диск с кодом доступа(acess code): y2ch [Google Drive](https://drive.google.com/open?id=1lsQS8hOCquMFKJFhK_z-n03ixWGkjT2P) | [Baidu Cloud](https://pan.baidu.com/s/1MsT3-He3UGipKhUi4OcCJw) 

1. Поместите предварительно-обученную(pre-trained) photo2cartoon-модель **photo2cartoon_weights.pt** в каталог моделей `models` (обновление 4 мая 2020).
2. Поместите предварительно-обученную(pre-trained)  модель сегментации головы  **seg_model_384.pb**  в каталог утилит `utils` . 
3. Поместите предварительно-обученную(pre-trained)  модель распознавания лиц **model_mobilefacenet.pth** в каталог моделей `models`   (Из [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)).
4. Набор данных мульт-рисунков с открытым исходным кодом **`cartoon_data/`** содержит `trainB` и `testB`.
5. Поместите  photo2cartoon onnx-модель весов **photo2cartoon_weights.onnx** [Google Drive](https://drive.google.com/file/d/1PhwKDUhiq8p-UqrfHCqj257QnqBWD523/view?usp=sharing) в каталог моделей `models`.

### Тестирование

Используйте фотографию  молодой азиатской женщины.
```
python test.py --photo_path ./images/photo_test.jpg --save_path ./images/cartoon_result.png
```

### Тестирование onnx-модели
```
python test_onnx.py --photo_path ./images/photo_test.jpg --save_path ./images/cartoon_result.png
```

### Обучение(тренировка)
**1.Data**

Данные для обучения содержат фотографии в видепортретов(домен A) и мультяшные рисунки (домен B). Следующий процесс может помочь уменьшиь трудность оптимизации.
- Обнаружить лицо(face) и его признаки-ориентиры(landmarks).
- Выравнивание лица(face) по его признакам-ориентирам(landmarks).
- разверните bbox признаков-ориентиров(landmarks) и обрежьте лицо(crop face).
- удалите фон семантическим сегментом(background by semantic segment).

<div align='center'>
  <img src='./images/data_process.jpg'>
</div>

Мы предоставляем 204 мультяшных изображения, кроме того, вы должны подготовить приблизительно 1,000 фотографий молодых азиатских женщин и предварительно обработать их следующей командой.

```
python data_process.py --data_path YourPhotoFolderPath --save_path YourSaveFolderPath
```

Каталог набора данных `dataset`  должен быть похож на эту структуру:
```
├── dataset
    └── photo2cartoon
        ├── trainA
            ├── xxx.jpg
            ├── yyy.png
            └── ...
        ├── trainB
            ├── zzz.jpg
            ├── www.png
            └── ...
        ├── testA
            ├── aaa.jpg 
            ├── bbb.png
            └── ...
        └── testB
            ├── ccc.jpg 
            ├── ddd.png
            └── ...
```

**2. Обучение(Train)**

Обучение с ноля:
```
python train.py --dataset photo2cartoon
```

Загрузите предварительно-обученные веса(pre-trained weights):
```
python train.py --dataset photo2cartoon --pretrained_weights models/photo2cartoon_weights.pt
```

Обучение с Multi-GPU:
```
python train.py --dataset photo2cartoon --batch_size 4 --gpu_ids 0 1 2 3
```

## Вопросы и ответы
#### Вопрос：Почему результат этого проекта, отличающегося от мини-программы?

Ответ: Для лучшей производительности мы настроили мультяшные данные (приблизительно 200 изображений), как у обученной модели для мини-программы. Мы также улучшили входной размер для высокой четкости. Кроме того, мы приняли нашу внутреннюю модель распознавания, чтобы вычислить Потерю ID Лица (Face ID Loss), которая намного лучше, чем в методе с открытыми исходниками, используемом в этом репозитарии(repo).

#### Вопрос: Как выбрать лучшую модель?

Ответ: Мы обучали модель за 200k итераций, затем выбрали лучшую модель согласно метрике признаков лиц FID(FID metric).

#### Вопрос: О модели распознавания лиц.

Ответ: Мы решили, что результат эксперимента вычисления Потери ID Лица (Face ID Loss) на нашей внутренней модели распознавания намного лучше, чем в методе с открытыми исходниками. Вы можете попытаться удалить вычисление Потери ID Лица (Face ID Loss) , если результат нестабилен.

#### Вопрос：Могу ли я использовать модель сегментации(segmentation model), чтобы предсказать поясной портрет(half-length portrait)?
Ответ：Нет. Модель обуччена специально для  для обрезанного лица.

## Литература

U-GAT-IT: Неконтолируемые Порождающие Соперничающие Сети (Unsupervised Generative Attentional Networks ) с  нормализацией адаптивного экземпляра-слоя(Adaptive Layer-Instance Normalization) для перевода изображения в  изображение(Image-to-Image Translation)  [Документ: [Paper](https://arxiv.org/abs/1907.10830)][Код: [Code](https://github.com/znxlwm/UGATIT-pytorch)]

Понимание лиц в Pytorch [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
