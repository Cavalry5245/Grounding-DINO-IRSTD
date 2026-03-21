# 消融实验结果

## Group 0

| Experiment   | Description      | HF-LoRA   | HF-Modules   | Prompt Bank   | Prompt Cats   |        P |        R |       F1 |   mAP@0.5 |
|:-------------|:-----------------|:----------|:-------------|:--------------|:--------------|---------:|---------:|---------:|----------:|
| Exp-0.1      | Full Fine-tuning | No        | -            | No            | -             | 0.175699 | 0.354305 | 0.234908 |  0.129998 |
| Exp-0.2      | Standard LoRA    | No        | -            | No            | -             | 0.968092 | 0.907285 | 0.936702 |  0.977085 |

## Group 1

| Experiment   | Description        | HF-LoRA   | HF-Modules   | Prompt Bank   | Prompt Cats   |        P |        R |       F1 |   mAP@0.5 |
|:-------------|:-------------------|:----------|:-------------|:--------------|:--------------|---------:|---------:|---------:|----------:|
| Exp-1.1      | HF-LoRA (qkv only) | Yes       | qkv          | No            | -             | 0.97826  | 0.893993 | 0.93423  |  0.974761 |
| Exp-1.2      | HF-LoRA (fc1 only) | Yes       | fc1          | No            | -             | 0.96755  | 0.897351 | 0.931129 |  0.974765 |
| Exp-1.3      | HF-LoRA (fc2 only) | Yes       | fc2          | No            | -             | 0.978339 | 0.897338 | 0.936089 |  0.976344 |
| Exp-1.4      | HF-LoRA (qkv+fc1)  | Yes       | qkv+fc1      | No            | -             | 0.951547 | 0.9104   | 0.930519 |  0.976617 |
| Exp-1.5      | HF-LoRA (Full)     | Yes       | qkv+fc1+fc2  | No            | -             | 0.96783  | 0.897351 | 0.931259 |  0.975577 |

## Group 2

| Experiment   | Description           | HF-LoRA   | HF-Modules   | Prompt Bank   | Prompt Cats   |        P |        R |       F1 |   mAP@0.5 |
|:-------------|:----------------------|:----------|:-------------|:--------------|:--------------|---------:|---------:|---------:|----------:|
| Exp-2.1      | Prompt Bank (Generic) | No        | -            | Yes           | generic       | 0.971309 | 0.896798 | 0.932568 |  0.977733 |
| Exp-2.2      | Prompt Bank (Gen+App) | No        | -            | Yes           | gen+app       | 0.967856 | 0.903974 | 0.934825 |  0.978163 |
| Exp-2.3      | Prompt Bank (Gen+Phy) | No        | -            | Yes           | gen+phy       | 0.964784 | 0.907162 | 0.935086 |  0.976536 |
| Exp-2.4      | Prompt Bank (3 cat)   | No        | -            | Yes           | gen+app+phy   | 0.963766 | 0.89404  | 0.927595 |  0.976781 |
| Exp-2.5      | Prompt Bank (5 cat)   | No        | -            | Yes           | all 5         | 0.954538 | 0.903811 | 0.928482 |  0.976592 |

## Group 3

| Experiment   | Description              | HF-LoRA   | HF-Modules   | Prompt Bank   | Prompt Cats   |        P |        R |       F1 |   mAP@0.5 |
|:-------------|:-------------------------|:----------|:-------------|:--------------|:--------------|---------:|---------:|---------:|----------:|
| Exp-3.1      | HF-LoRA + Prompt (3 cat) | Yes       | qkv+fc1+fc2  | Yes           | gen+app+phy   | 0.971518 | 0.903584 | 0.93632  |  0.978564 |
| Exp-3.2      | HF-LoRA + Prompt (5 cat) | Yes       | qkv+fc1+fc2  | Yes           | all 5         | 0.95467  | 0.907285 | 0.930375 |  0.975945 |

## Group 4

| Experiment   | Description      | HF-LoRA   | HF-Modules   | Prompt Bank   | Prompt Cats   |        P |        R |       F1 |   mAP@0.5 |
|:-------------|:-----------------|:----------|:-------------|:--------------|:--------------|---------:|---------:|---------:|----------:|
| Exp-4.1      | HF-LoRA r=8      | Yes       | qkv+fc1+fc2  | Yes           | gen+app+phy   | 0.967965 | 0.900462 | 0.932994 |  0.976481 |
| Exp-4.2      | HF-LoRA r=32     | Yes       | qkv+fc1+fc2  | Yes           | gen+app+phy   | 0.951868 | 0.91677  | 0.933989 |  0.977177 |
| Exp-4.3      | HF-LoRA alpha=16 | Yes       | qkv+fc1+fc2  | Yes           | gen+app+phy   | 0.954534 | 0.903741 | 0.928443 |  0.97553  |
| Exp-4.4      | HF-LoRA alpha=64 | Yes       | qkv+fc1+fc2  | Yes           | gen+app+phy   | 0.978257 | 0.893868 | 0.93416  |  0.977105 |

