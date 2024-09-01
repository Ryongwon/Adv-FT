# Adv-FT

| Title | Years | Problem | Method | Link | Quote |
| --- | --- | --- | --- | --- | --- |
| Using Pre-Training Can Improve Model Robustness and Uncertainty | 2019 | 预训练模型与重新训练模型效果相同，预训练的作用在哪？ | 提出预训练可以提高不确定性估计，对抗性预训练可以提高模型的鲁棒性 | [code](https://github.com/hendrycks/pre-training) | [paper](https://arxiv.org/pdf/1901.09960) |
| Theoretically Principled Trade-off between Robustness and Accuracy | 2019 | 如何权衡对抗鲁棒性与自然准确性 | 将鲁棒性分解为自然图像准确性和临界区域准确性，通过权重使鲁棒性最接近准确性 | [code](https://github.com/yaodongyu/TRADES) | [paper](https://arxiv.org/pdf/1901.08573) |
| A Simple Fine-tuning Is All You Need: Towards Robust Deep Learning Via Adversarial Fine-tuning | 2020 | 基于PGD的对抗训练计算成本高且容易过拟合 | 用慢启动快衰减的方式调整微调任务的学习率，从而降低成本并保证泛化性 | [code]() | [paper](https://arxiv.org/pdf/2012.13628) |
| LAS-AT: Adversarial Training with Learnable Attack Strategy | 2022 | 固定的攻击参数限制了模型鲁棒性上限 | 通过类似GAN的方式动态学习生成攻击参数 | [code](https://github.com/jiaxiaojunQAQ/LAS-AT) | [paper](https://arxiv.org/pdf/2203.06616) |
| TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization | 2023 | 微调过程如何保证预训练模型的鲁棒性 | 通过GAN的方式，部分网络保留预训练模型鲁棒性，另一部分网络更新微调任务鲁棒性 | [code](https://github.com/ziquanliu/CVPR2023-TWINS) | [paper](https://arxiv.org/pdf/2303.11135) |
| Improving Generalization of Adversarial Training via Robust Critical Fine-Tuning | 2023 | 如何在不影响对抗鲁棒性的情况下增强泛化 | 在鲁棒性影响最低的模块上微调对抗训练的模型来利用冗余容量来实现鲁棒性 | [code](https://github.com/microsoft/robustlearn) | [paper](https://arxiv.org/pdf/2308.02533) |
| ASAM: Boosting Segment Anything Model with Adversarial Tuning | 2024 | 预训练的图像分割模型SAM在特定微调表现较差 | 生成更真实的对抗性样本对该模型进行微调 | [code]() | [paper](https://arxiv.org/pdf/2405.00256) |
| FullLoRA-AT: Efficiently Boosting the Robustness of Pretrained Vision Transformers | 2024 | 如何使用少量的附加参数进行对抗性微调 | 添加可学习的BN层，同时将该模块融入到预训练模型形成框架，更好进行参数高效化 | [code](https://github.com/luckybird1994/ASAM) | [paper](https://arxiv.org/pdf/2401.01752) |
| One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models | 2024 | 不通过冻结权重的方式，如何提高预训练模型的鲁棒性 | 提出以文本提示（Prompt）的方式进行微调，提示参数化的同时并进行动态调整，从而提高鲁棒性并降低计算成本 | [code](https://github.com/TreeLLi/APT) | [paper](https://arxiv.org/pdf/2403.01849) |
| Adversarial Prompt Tuning for Vision-Language Models | 2024 |  |  | [code](https://github.com/jiamingzhang94/Adversarial-Prompt-Tuning) | [paper](https://arxiv.org/pdf/2311.11261) |
| Securely Fine-Tuning Pre-Trained Encoders Against Adversarial Examples | 2024 |  |  | [code]() | [paper](https://arxiv.org/pdf/2403.10801) |
| Sim-CLIP: Unsupervised Siamese Adversarial Fine-Tuning for Robust and Semantically-Rich Vision-Language Models | 2024 |  |  | [code]() | [paper](https://arxiv.org/pdf/2407.14971) |
| AutoLoRa: An Automated Robust Fine-Tuning Framework | 2024 | 如何解决提高鲁棒性的微调中，自然样本与对抗样本梯度优化方向不一致 | 在优化自然样本时引入LoRA分支，优化对抗样本时使用FE分支，以解耦的方式解决梯度优化方向不一致；同时使用FE分支时能减少下游任务的超参数，节省计算成本 | [code](https://github.com/GodXuxilie/RobustSSL_Benchmark/tree/main/Finetuning_Methods/AutoLoRa) | [paper](https://arxiv.org/pdf/2310.01818) |
