# DFFG-MLLMR
This is the code repository of the paper "DFFG-MLLMR: Dynamic Feature Fusion Guiding and Multimodal Large Language Model Refining for Medical Image Report Generation"
![image][DFFG-MLLMR/1.png](https://github.com/BearLiX/DFFG-MLLMR/blob/main/DFFG-MLLMR/1.png)

## Qwen2-vl
You can get the original weights of LLM here "https://github.com/QwenLM/Qwen2-VL"
You can find the files for fine-tuning LLM in .\llm
It is recommended to use the following tools for fine-tuning "https://github.com/hiyouga/LLaMA-Factory"

## medclip
You can find the raw weights of the visual encoder at https://github.com/RyanWangZf/MedCLIP

## Datasets
The medical image report generation datasets are available at the following links:
1. MIMIC-CXR-JPG data can be found at https://physionet.org/content/mimic-cxr-jpg/2.0.0/.
2. IU X-Ray data can be found at https://openi.nlm.nih.gov/.

### Training

To train the model, you need to prepare the training dataset. For example, the IU X-Ray data.
Check the dataset path in train.py, and then run:
```
python train.py
```
### Testing
Check the model and data path in test.py, and then run:
```
python test.py
```

# the metric meteor
the paraphrase-en.gz should be put into the .\pycocoevalcap\meteor\data, since the file is too big to upload.
