<div align="center">
<p>
   <a align="center" target="_blank">
   <img width="900" src="./images/MiVOLO.jpg"></a>
</p>
<br>
</div>



## MiVOLO: Multi-input Transformer for Age and Gender Estimation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-estimation-on-utkface)](https://paperswithcode.com/sota/age-estimation-on-utkface?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-estimation-on-imdb-clean)](https://paperswithcode.com/sota/age-estimation-on-imdb-clean?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/facial-attribute-classification-on-fairface)](https://paperswithcode.com/sota/facial-attribute-classification-on-fairface?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-and-gender-classification-on-adience)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-and-gender-classification-on-adience-age)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-and-gender-estimation-on-lagenda-age)](https://paperswithcode.com/sota/age-and-gender-estimation-on-lagenda-age?p=mivolo-multi-input-transformer-for-age-and) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mivolo-multi-input-transformer-for-age-and/age-and-gender-estimation-on-lagenda-gender)](https://paperswithcode.com/sota/age-and-gender-estimation-on-lagenda-gender?p=mivolo-multi-input-transformer-for-age-and)

> [**MiVOLO: Multi-input Transformer for Age and Gender Estimation**](https://arxiv.org/abs/2307.04616),
> Maksim Kuprashevich, Irina Tolstykh,
> *2023 [arXiv 2307.04616](https://arxiv.org/abs/2307.04616)*

[[`Paper`](https://arxiv.org/abs/2307.04616)] [[`Demo`](https://huggingface.co/spaces/iitolstykh/age_gender_estimation_demo)] [[`BibTex`](#citing)] [[`Data`](https://wildchlamydia.github.io/lagenda/)]


## MiVOLO pretrained models

Gender & Age recognition performance.

<table style="margin: auto">
  <tr>
    <th align="left">Model</th>
    <th align="left" style="color:LightBlue">Type</th>
    <th align="left">Dataset</th>
    <th align="left">Age MAE</th>
    <th align="left">Age CS@5</th>
    <th align="left">Gender Accuracy</th>
    <th align="left">download</th>
  </tr>
  <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.29</td>
    <td align="left">67.71</td>
    <td align="left">-</td>
    <td><a href="https://drive.google.com/file/d/17ysOqgG3FUyEuxrV3Uh49EpmuOiGDxrq/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age, gender</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.22</td>
    <td align="left">68.68</td>
    <td align="left">99.38</td>
    <td><a href="https://drive.google.com/file/d/1NlsNEVijX2tjMe8LBb1rI56WB_ADVHeP/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">IMDB-cleaned</td>
    <td align="left">4.24 [face+body]<br>6.87 [body]</td>
    <td align="left">68.32 [face+body]<br>46.32 [body]</td>
    <td align="left">99.46 [face+body]<br>96.48 [body]</td>
    <td><a href="https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view?usp=drive_link">checkpoint</a></td>
  </tr>
  <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age</td>
    <td align="left">UTKFace</td>
    <td align="left">4.23</td>
    <td align="left">69.72</td>
    <td align="left">-</td>
    <td><a href="https://drive.google.com/file/d/1LtDfAJrWrw-QA9U5IuC3_JImbvAQhrJE/view?usp=drive_link">checkpoint</a></td>
  </tr>
    <tr>
    <td>volo_d1</td>
    <td align="left">face_only, age, gender</td>
    <td align="left">UTKFace</td>
    <td align="left">4.23</td>
    <td align="left">69.78</td>
    <td align="left">97.69</td>
    <td><a href="https://drive.google.com/file/d/1hKFmIR6fjHMevm-a9uPEAkDLrTAh-W4D/view?usp=drive_link">checkpoint</a></td>
  </tr>
  <tr>
    <td>mivolo_d1</td>
    <td align="left">face_body, age, gender</td>
    <td align="left">Lagenda</td>
    <td align="left">3.99 [face+body]</td>
    <td align="left">71.27 [face+body]</td>
    <td align="left">97.36 [face+body]</td>
    <td><a href="https://huggingface.co/spaces/iitolstykh/demo">demo</a></td>
  </tr>
<tr>

</table>


## Dataset

**Please, [cite our paper](#citing) if you use any this data!**

- Lagenda dataset: [images](https://drive.google.com/file/d/1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG/view?usp=sharing) and [annotation](https://drive.google.com/file/d/1mNYjYFb3MuKg-OL1UISoYsKObMUllbJx/view?usp=sharing).
- IMDB-clean: follow [these instructions](https://github.com/yiminglin-ai/imdb-clean) to get images and [download](https://drive.google.com/file/d/17uEqyU3uQ5trWZ5vRJKzh41yeuDe5hyL/view?usp=sharing) our annotations.
- UTK dataset: [origin full images](https://susanqq.github.io/UTKFace/) and our annotation: [split from the article](https://drive.google.com/file/d/1Fo1vPWrKtC5bPtnnVWNTdD4ZTKRXL9kv/view?usp=sharing), [random full split](https://drive.google.com/file/d/177AV631C3SIfi5nrmZA8CEihIt29cznJ/view?usp=sharing).
- Adience dataset: follow [these instructions](https://talhassner.github.io/home/projects/Adience/Adience-data.html) to get images and [download](https://drive.google.com/file/d/1wS1Q4FpksxnCR88A1tGLsLIr91xHwcVv/view?usp=sharing) our annotations.
   <details>
      <summary>Click to expand!</summary>

   After downloading them, your `data` directory should look something like this:

   ```console
   data
   └── Adience
       ├── annotations  (folder with our annotations)
       ├── aligned      (will not be used)
       ├── faces
       ├── fold_0_data.txt
       ├── fold_1_data.txt
       ├── fold_2_data.txt
       ├── fold_3_data.txt
       └── fold_4_data.txt
   ```

   We use coarse aligned images from `faces/` dir.

   Using our detector we found a face bbox for each image (see [tools/prepare_adience.py](tools/prepare_adience.py)).

   This dataset has five folds. The performance metric is accuracy on five-fold cross validation.

   | images before removal | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
   | --------------------- | ------ | ------ | ------ | ------ | ------ |
   | 19,370                | 4,484  | 3,730  | 3,894  | 3,446  | 3,816  |

   Not complete data

   | only age not found | only gender not found | SUM           |
   | ------------------ | --------------------- | ------------- |
   | 40                 | 1170                  | 1,210 (6.2 %) |

   Removed data

   | failed to process image | age and gender not found | SUM         |
   | ----------------------- | ------------------------ | ----------- |
   | 0                       | 708                      | 708 (3.6 %) |

   Genders

   | female | male  |
   | ------ | ----- |
   | 9,372  | 8,120 |

   Ages (8 classes) after mapping to not intersected ages intervals

   | 0-2   | 4-6   | 8-12  | 15-20 | 25-32 | 38-43 | 48-53 | 60-100 |
   | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ |
   | 2,509 | 2,140 | 2,293 | 1,791 | 5,589 | 2,490 | 909   | 901    |

   </details>

- FairFace dataset: follow [these instructions](https://github.com/joojs/fairface) to get images and [download](https://drive.google.com/file/d/1EdY30A1SQmox96Y39VhBxdgALYhbkzdm/view?usp=drive_link) our annotations.
    <details>
      <summary>Click to expand!</summary>

    After downloading them, your `data` directory should look something like this:

    ```console
    data
    └── FairFace
       ├── annotations  (folder with our annotations)
       ├── fairface-img-margin025-trainval   (will not be used)
           ├── train
           ├── val
       ├── fairface-img-margin125-trainval
           ├── train
           ├── val
       ├── fairface_label_train.csv
       ├── fairface_label_val.csv

    ```

    We use aligned images from `fairface-img-margin125-trainval/` dir.

    Using our detector we found a face bbox for each image and added a person bbox if it was possible (see [tools/prepare_fairface.py](tools/prepare_fairface.py)).

    This dataset has 2 splits: train and val. The performance metric is accuracy on validation.

    | images train | images val |
    | ------------ | ---------- |
    | 86,744       | 10,954     |

    Genders for **validation**

    | female | male  |
    | ------ | ----- |
    | 5,162  | 5,792 |

    Ages for **validation** (9 classes):

    | 0-2 | 3-9   | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+ |
    | --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --- |
    | 199 | 1,356 | 1,181 | 3,300 | 2,330 | 1,353 | 796   | 321   | 118 |

    </details>

## Install

Install pytorch 1.13+ and other requirements.

```
pip install -r requirements.txt
pip install .
```


## Demo

1. [Download](https://drive.google.com/file/d/1CGNCkZQNj5WkP3rLpENWAOgrBQkUWRdw/view) body + face detector model to `models/yolov8x_person_face.pt`
2. [Download](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) mivolo checkpoint to `models/mivolo_imbd.pth.tar`

```bash
wget https://variety.com/wp-content/uploads/2023/04/MCDNOHA_SP001.jpg -O jennifer_lawrence.jpg

python3 demo.py \
--input "jennifer_lawrence.jpg" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt " \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--with-persons \
--draw
```

To run demo for a youtube video:
```bash
python3 demo.py \
--input "https://www.youtube.com/shorts/pVh32k0hGEI" \
--output "output" \
--detector-weights "models/yolov8x_person_face.pt" \
--checkpoint "models/mivolo_imbd.pth.tar" \
--device "cuda:0" \
--draw \
--with-persons
```


## Validation

To reproduce validation metrics:

1. Download prepared annotations for imbd-clean / utk / adience / lagenda  / fairface.
2. Download checkpoint
3. Run validation:

```bash
python3 eval_pretrained.py \
  --dataset_images /path/to/dataset/utk/images \
  --dataset_annotations /path/to/dataset/utk/annotation \
  --dataset_name utk \
  --split valid \
  --batch-size 512 \
  --checkpoint models/mivolo_imbd.pth.tar \
  --half \
  --with-persons \
  --device "cuda:0"
````

Supported dataset names: "utk", "imdb", "lagenda", "fairface", "adience".


## Changelog

15.08.20223 - 0.4.1dev

### Added
- Support for video streams, including YouTube URLs
- Instructions and explanations for various export types.

### Changed
- Removed CutOff operation. It has been proven to be ineffective for inference time and quite costly at the same time. Now it is only used during training.
  
## ONNX and TensorRT export

As of now (11.08.2023), while ONNX export is technically feasible, it is not advisable due to the poor performance of the resulting model with batch processing. 
**TensorRT** and **OpenVINO** export is impossible due to its lack of support for col2im.

If you remain absolutely committed to utilizing ONNX export, you can refer to [these instructions](https://github.com/WildChlamydia/MiVOLO/issues/14#issuecomment-1675245889). 

The most highly recommended export method at present **is using TorchScript**. You can achieve this with a single line of code:
```python
torch.jit.trace(model)
```
This approach provides you with a model that maintains its original speed and only requires a single file for usage, eliminating the need for additional code.

## License

Please, see [here](./license)


## Citing

If you use our models, code or dataset, we kindly request you to cite the following paper and give repository a :star:

```bibtex
@article{mivolo2023,
   Author = {Maksim Kuprashevich and Irina Tolstykh},
   Title = {MiVOLO: Multi-input Transformer for Age and Gender Estimation},
   Year = {2023},
   Eprint = {arXiv:2307.04616},
}
```
