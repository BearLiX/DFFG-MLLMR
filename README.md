# DFFG-MLLMR
This is the code repository of the paper "DFFG-MLLMR"

![image](https://github.com/BearLiX/DFFG-MLLMR/blob/main/DFFG-MLLMR/1.png)

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

# prompt
Because identifying an “optimal” prompt is inherently experimental and demands repeated trials and adjustments, we have added the schematic below to clarify how each prompt in our study was conceived and refined. Our design philosophy is grounded in recent literature and tailored specifically to the chest‑X‑ray diagnostic setting. It proceeds in an iterative loop: design the prompt → refine the prompt → test the revised prompt → analyze bad cases → refine the prompt again. Within this loop, every prompt is decomposed into five elements: context, goal, style, audience, and output. Context conveys task‑specific background so the LLM reasons within the correct clinical environment. Goal is to clearly point out the specific tasks we expect LLM to complete and guide LLM to focus on the current task by setting clear and precise goal instructions. Style constrains the narrative to resemble authentic radiology reports. Audience identifies the intended readership, enabling the model to adjust language depth appropriately. Output stipulates the final text format, ensuring that the model returns a usable diagnostic report rather than, for example, a list or JSON.
<img width="865" height="537" alt="image" src="https://github.com/user-attachments/assets/4810d952-8a82-4aea-94c8-e680f664f1cb" />
Taking the baseline prompt employed with our foundation model Qwen‑7B (Qwen2-VL-7B-Instruct) as an example, the table below shows how each sentence fragment aligns with the five design components—context, goal, style, audience, and output.
<img width="1186" height="886" alt="image" src="https://github.com/user-attachments/assets/607f7f87-7c65-4157-8d0e-40946f370413" />
To enable a clear, side‑by‑side comparison, Table 3 presents the complete prompts used for every model. The segment that departs from the baseline prompt is highlighted in yellow. In the discussion that follows, we examine these changes individually and explain the specific role each highlighted phrase plays. For the fine‑tuned model (Qwen‑7B‑FT), which was further trained on the IU‑Xray and MIMIC‑CXR corpora, we removed the clause “Output strictly in English, and do not include formatting or provide additional medical advice.” Because the fine‑tuning data consist exclusively of English radiology reports and the model now consistently produces plain, unformatted prose, this instruction has become unnecessary. Conversely, both datasets anonymize protected health information with placeholders such as “XXXX” . To keep the model from mistaking these tokens for clinical content or copying them into its output, we added the directive “Patient reports may have privacy processing, do not pay attention.” 
For the retrieval‑augmented model （Qwen‑7B‑FT + Visual Retrieve）, the prompt must explain the role of the extra text. We add the sentence: “To make the style of the diagnosis match a real report, we provide a template drawn from images similar to the current chest X‑ray. If you encounter similar wording, use the template to polish your own text. The retrieved report: {retrieved report}.” We added the directive that labels the retrieved report as a stylistic template from similar cases, instructs the model to consult it only for wording, prevents it from treating the text as information about the current pathological image itself, and thus aligns the generated diagnosis with authentic radiology style while preserving case‑specific details.
In the DFFG‑MLLMR stage, the LLM stops being a report generator and turns into a multi‑tasker. It now has to receive three things at once—the raw chest X‑ray, the draft report coming out of the DFFG module, and a report provided by visual retrieval. The prompt words in this stage achieve three goals: (1) the LLM report is now generated based on the report of the DFFG module, which can reduce the risk of hallucination and redundant wording of the model; (2) the reports returned by visual retrieval are used as language and diagnosis references, thereby improving style fidelity and clinical accuracy; (3) the task is defined as iterative rewriting rather than complete reproduction, while retaining the model's diagnostic findings on pathological images.
<img width="906" height="908" alt="image" src="https://github.com/user-attachments/assets/0cbcddbf-ce66-444d-bc21-0771709e3a71" />
