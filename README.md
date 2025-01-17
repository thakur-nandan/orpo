# **ORPO**

This is the official repository for <a class="link" href="https://arxiv.org/abs/2403.07691">**ORPO: Monolithic Preference Optimization without Reference Model**</a>. 

<details>
<summary><b>UPDATES (2023.03.18)</b></summary>
<div markdown="1">

- [X] FSDP model saving error: Resolved checkpoint saving error which originates from the conflict between FSDP and `torch.compile`
- [X] Now **Mistral-ORPO-β** is in the <a class="link" href="https://tatsu-lab.github.io/alpaca_eval/">official AlpacaEval Leaderboard!</a>

</div>
</details>
&nbsp;

The detailed results in the paper can be found in:
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?query=kaist-ai%2Fmistral-orpo-beta)
- [AlpacaEval](#alpacaeval)
- [MT-Bench](#mt-bench)
- [IFEval](#ifeval)

&nbsp;

### **`Model Checkpoints`**

Our models trained with ORPO can be found in:

- [X] **Mistral-ORPO-⍺**: 🤗 <a class="link" href="https://huggingface.co/kaist-ai/mistral-orpo-alpha">kaist-ai/mistral-orpo-alpha</a>
- [X] **Mistral-ORPO-β**: 🤗 <a class="link" href="https://huggingface.co/kaist-ai/mistral-orpo-beta">kaist-ai/mistral-orpo-beta</a>

And the corresponding logs for the average log probabilities of chosen/rejected responses during training are reported in:

- [X] **Mistral-ORPO-⍺**: <a class="link" href="https://wandb.ai/jiwooya1000/PREF/reports/Mistral-ORPO-7B-Training-Log--Vmlldzo3MTE1NzE0?accessToken=rms6o4mg5vo3feu1bvbpk632m4cspe19l0u1p4he3othx5bgean82chn9neiile6">Wandb Report for Mistral-ORPO-⍺</a>
- [X] **Mistral-ORPO-β**: <a class="link" href="https://wandb.ai/jiwooya1000/PREF/reports/Mistral-ORPO-7B-Training-Log--Vmlldzo3MTE3MzMy?accessToken=dij4qbp6dcrofsanzbgobjsne9el8a2zkly2u5z82rxisd4wiwv1rhp0s2dub11e">Wandb Report for Mistral-ORPO-β</a>

&nbsp;

### **`AlpacaEval`**

<figure>
  <img class="png" src="/assets/img/alpaca_blog.png" alt="Description of the image">
  <figcaption><b>Figure 1.</b> AlpacaEval 2.0 score for the models trained with different alignment methods.</figcaption>
</figure>

&nbsp;

### **`MT-Bench`**

<figure>
  <img class="png" src="/assets/img/mtbench_hf.png" alt="Description of the image">
  <figcaption><b>Figure 2.</b> MT-Bench result by category.</figcaption>
</figure>

&nbsp;

### **`IFEval`**

IFEval scores are measured with <a class="link" href="https://github.com/EleutherAI/lm-evaluation-harness">EleutherAI/lm-evaluation-harness</a> by applying the chat template. The scores for Llama-2-Chat (70B), Zephyr-β (7B), and Mixtral-8X7B-Instruct-v0.1 are originally reported in <a class="link" href="https://twitter.com/wiskojo/status/1739767758462877823">this tweet</a>.

| **Model Type**     | **Prompt-Strict** | **Prompt-Loose** | **Inst-Strict** | **Inst-Loose** |
|--------------------|:-----------------:|:----------------:|:---------------:|----------------|
| **Llama-2-Chat (70B)** |       0.4436      |      0.5342      |      0.5468     |     0.6319     |
| **Zephyr-β (7B)** |       0.4233      |      0.4547      |      0.5492     |     0.5767     |
| **Mixtral-8X7B-Instruct-v0.1** |       0.5213      |      **0.5712**      |      0.6343     |     **0.6823**     |
| **Mistral-ORPO-⍺ (7B)** |       0.5009      |      0.5083      |      0.5995     |     0.6163     |
| **Mistral-ORPO-β (7B)** |       **0.5287**      |      0.5564      |      **0.6355**     |     0.6619     |
