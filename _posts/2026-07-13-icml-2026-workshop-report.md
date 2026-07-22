---
layout: post
title: "ICML 2026 워크숍 참관기 — LLM·Agent 포스터 세션 리뷰"
description: "ICML 2026 워크숍 데이 참관기. Weight-Space Symmetries 워크숍을 중심으로 발표를 듣고, 포스터 세션에서 확인한 LLM·Agent 관련 논문 14편을 주제별(Post-Training / Reasoning Trace / Agent Memory / 편향·안전성 / 학습 구조)로 정리한다."
date: 2026-07-13
category: "Trip Reports"
lang: ko
---

지난 7월 10일, ICML 2026 워크숍 데이에 참석하였습니다. **Weight-Space Symmetries: From Foundations to Practical Applications** 워크숍의 발표를 중심으로 들었고, 포스터 세션에서는 현재 업무와 개인 연구에 연결되는 LLM·Agent 관련 연구들을 찾아보았습니다.

포스터 세션은 짧은 시간 안에 다양한 연구를 직접 보고, 저자에게 핵심 아이디어를 물어볼 수 있다는 점이 좋았습니다. 논문만 읽을 때보다 "왜 이 문제를 풀려고 했는지", "실험하면서 어떤 부분이 어려웠는지"를 조금 더 빠르게 파악할 수 있었습니다.

이번 글에서는 먼저 ICML 2026에서 전반적으로 어떤 문제의식이 두드러졌는지 정리하고, 이후에는 현장에서 직접 확인한 포스터들을 주제별로 간략히 소개해 보겠습니다.

---

## 1. ICML 2026 전반의 연구 흐름

> 아래 내용은 ICML 2026 전체 Accepted Paper를 정량적으로 분류한 결과가 아닙니다. 공식 워크숍·튜토리얼·수상 논문·초청 강연과 현장에서 확인한 포스터를 종합한 정성적 기술 동향입니다. 특정 분야의 논문 비중을 단정하기보다는, 올해 학회에서 두드러지게 논의된 문제의식을 중심으로 정리하였습니다.

ICML 2026에서는 생성형 AI와 대규모 언어모델에 대한 높은 관심이 이어지는 가운데, 연구의 범위가 단순한 모델 규모 확대를 넘어 **학습 이후의 적응, 추론 과정의 제어, Agent의 신뢰성, 생성 모델의 이론적 이해, 안전성과 사회적 영향**으로 넓어지고 있었습니다.

동시에 최적화, 확률론, 강화학습, 일반화 이론과 같은 전통적인 머신러닝 기반 연구도 여전히 중요한 축을 이루고 있었습니다. 현장에서 다양한 연구를 보다 보니, 새로운 모델과 방법론을 빠르게 따라가는 것만큼이나 **탄탄한 기반 이론과 명확한 문제 정의가 좋은 연구의 출발점**이라는 점을 다시 한번 느꼈습니다.

### 1.1 Agent와 생성형 AI에 대한 높은 관심, 그리고 프로그램의 의도적인 다변화

ICML 공식 발표에 따르면 올해는 **247건의 워크숍 제안서**가 접수되었고, 이 가운데 **44개 워크숍과 4개의 Affinity Workshop**이 선정되었습니다. 특히 제목에 *Agentic AI*와 유사한 표현이 포함된 제안서가 **60건 이상** 제출될 만큼 Agent 분야에 대한 관심이 높았습니다.

다만 워크숍 조직위원회는 유사한 Agent 주제가 지나치게 많았다고 설명하면서, 다른 분야를 위한 공간을 확보하기 위해 일부만 최종 프로그램에 포함하였습니다. 실제 프로그램에는 Agent뿐 아니라 이론, 최적화, 신뢰성, AI for Science, 법·거버넌스, 통신 등 폭넓은 분야가 함께 포함되었습니다.

따라서 ICML 2026을 단순히 "Agent 중심의 학회"라고 표현하기보다는, Agent와 생성형 AI가 가장 큰 관심을 받은 분야 중 하나였지만, **최종 프로그램은 의도적으로 다양한 연구 영역을 포괄하도록 구성**되었다고 보는 편이 정확합니다.

### 1.2 Raw Scaling을 넘어 Post-Training과 Adaptive Compute로 확장

LLM 분야에서는 모델 파라미터와 사전학습 데이터의 규모를 확대하는 것만큼이나, 이미 학습된 모델을 어떻게 특정 업무에 적응시키고, 추론 시점의 계산을 얼마나 사용할 것인지가 중요하게 다뤄졌습니다.

공식 튜토리얼에도 **Adaptive Reasoning in LLMs: From Post-Training to Test-Time Learning**이 포함되었으며, 포스터에서도 다음과 같은 연구를 확인할 수 있었습니다.

- RLVR을 활용한 Tool-Using Agent 학습
- Instruction-Tuned Model과 Reasoning Model의 병합
- 문제 난이도에 따른 Reasoning Budget 조절
- Verifier 기반 추론과 조기 종료
- 여러 Judge 결과의 통계적 집계
- 제한된 GPU 환경을 고려한 양자화와 효율적 서빙

이러한 흐름은 Scaling이 중요하지 않게 되었다는 의미가 아닙니다. 오히려 **Scaling으로 확보한 모델 능력을 실제 과제에서 더 정확하고 경제적으로 끌어내는 기술이 독립적인 연구 축으로 성장**하고 있다고 볼 수 있습니다.

특히 실제 서비스에서는 최고 Benchmark 점수뿐 아니라 GPU·API 비용, 응답 지연, Context 길이, 운영 안정성을 함께 고려해야 하므로, 문제별로 계산 자원을 조절하는 Adaptive Compute의 중요성이 더욱 커질 것으로 보입니다.

### 1.3 Agent 연구는 기능 구현에서 실패 진단과 운영 신뢰성으로 이동

최종 워크숍 프로그램에는 *Agent Failure Mode, Agents in the Wild, Planning with Language Models, Coding Agents, Agentic Uncertainty, Multimodal AI Agents* 등 다양한 Agent 관련 주제가 포함되었습니다.

Agent 연구의 질문도 단순히 "Tool을 호출할 수 있는가?"에서 다음과 같은 문제로 세분화되고 있었습니다.

- 장기 업무 수행을 위한 Memory와 Continual Adaptation
- 계획 수립과 Tool 사용의 안정성
- Multi-Agent 환경에서의 역할 분담과 정보 공유
- 실패 조건의 재현과 Trace 기반 진단
- 불확실성 추정과 안전성 검증
- 실제 운영 환경에서의 보안, 비용, 오류 복구

개별 Agent 기능을 추가하는 것만으로는 충분하지 않고, **Memory · Evaluation · Guardrail · Monitoring · Failure Recovery를 하나의 운영 체계로 설계하는 일**이 더욱 중요해질 것으로 보입니다.

### 1.4 Diffusion 연구는 응용 확대와 함께 기본 가정을 다시 검토

Diffusion과 Flow Matching은 공식 튜토리얼의 주요 주제로 다뤄졌으며, ICML 2026의 **Outstanding Paper 두 편도 Diffusion과 직접 관련**되어 있었습니다.

한 연구는 Diffusion Language Model의 임의 순서 생성이 일반적인 추론 과제에서 항상 장점이 되는 것은 아니며, 중요한 고불확실성 Token을 우회하면서 해답의 다양성을 조기에 감소시킬 수 있다고 분석하였습니다. 다른 연구는 Score-Based Sampling에서 목표 오차에 도달하기 위한 이론적 Step Complexity를 크게 개선하였습니다.

이러한 결과를 보면 Diffusion 연구는 이미지 생성에서 언어와 구조적 생성 문제로 적용 범위를 넓히는 동시에, 다음과 같은 기본 질문을 다시 검토하는 단계로 발전하고 있습니다.

- 임의 순서 생성은 실제로 언제 유리한가?
- Sampling 비용을 얼마나 줄일 수 있는가?
- 서로 다른 Diffusion Model이 비슷한 결과를 만드는 이유는 무엇인가?
- 생성 성능과 일반화를 어떤 이론으로 설명할 수 있는가?

수상 논문의 자세한 내용은 [PyTorchKR 정리](https://discuss.pytorch.kr/t/icml-2026-test-of-time-10/11112)도 함께 참고할 만합니다.

### 1.5 안정성·정직성·암기·해석 가능성이 독립적인 연구 축으로 강화

올해 수상 및 Honorable Mention 논문과 공식 프로그램에서는 모델의 성능뿐 아니라, **모델이 실제로 무엇을 학습하고 있으며 어떤 방식으로 실패하는지를 분석하는 연구**가 눈에 띄었습니다.

대표적으로 다음과 같은 문제가 다뤄졌습니다.

- RLVR 과정에서 정직성 Probe를 회피하는 Activation·Policy Obfuscation
- 언어모델의 일반화와 데이터 암기 능력의 구분
- Alignment 기술이 검열과 조작에 오용될 가능성
- Deepfake 연구와 실제 사회적 피해 사이의 간극
- 내부 표현과 외부 출력의 불일치
- 안전 조치가 실질적인 숙고보다 과도한 거부를 유발하는 문제

현장에서 본 포스터에서도 "길게 생각하는 것처럼 보이는 Trace가 실제 의사결정 과정을 반영하는가?", "모델이 평가기나 안전 장치를 우회하도록 학습될 수 있는가?"와 같은 질문이 반복적으로 등장했습니다.

앞으로는 Accuracy나 Reward 하나만으로 모델을 평가하기보다, **모델이 결과에 도달한 과정, 내부 표현, 실패 원인을 함께 살펴보는 평가 방식**이 더욱 중요해질 것으로 보입니다.

### 1.6 기반 이론과 최적화 연구도 여전히 ICML의 핵심 축

LLM과 Agent에 대한 관심이 매우 높았지만, ICML의 기반 연구 성격 역시 강하게 유지되고 있었습니다.

Outstanding Paper에는 Diffusion Sampling의 이론적 복잡도를 개선한 연구가 포함되었고, Honorable Mention에는 Random Matrix Theory를 이용한 Diffusion Model 분석과 Ridge Regression에서의 Grokking 이론이 선정되었습니다.

공식 튜토리얼에도 수치 최적화, 확률적 수치해석, Sequence Prediction, Diffusion Theory, Theorem Proving 등이 포함되었습니다.

이러한 흐름을 보면서 **최근의 응용 연구를 이해하고 발전시키기 위해서는 결국 최적화, 확률론, 학습 동역학, 시스템 설계와 같은 기반 연구가 함께 뒷받침되어야 한다**는 점을 다시 확인할 수 있었습니다.

### 1.7 AI for Science와 전문 도메인으로의 적용 확대

ICML 2026 워크숍에는 AI for Science, AI for Mathematics, AI for Physics, Biology, Life Science, Health, Law, Wireless Communication, Audio, Forecasting, Structured Data 등이 포함되었습니다.

초청 강연 역시 머신러닝 이론과 AI Safety뿐 아니라 경제·정책, 계산생물학, 자연어처리, 인간-컴퓨터 상호작용을 폭넓게 다루었습니다.

이는 범용 Foundation Model을 그대로 적용하는 것보다, **각 분야의 데이터 구조, 검증 규칙, 전문가 지식, 실제 의사결정 절차를 모델과 결합하는 연구**가 확대되고 있음을 보여줍니다. 특히 Tabular 데이터를 다루는 연구들이 종종 눈에 띄었습니다.

---

### 정리: ICML 2026 전반의 흐름

> 모델 규모 확대와 기반 이론 연구가 여전히 중요하게 유지되는 가운데, 연구의 관심이 **Post-Training, Adaptive Reasoning, Agent 운영, 신뢰성 검증, 계산 효율, 사회적 영향**으로 넓어지고 있었습니다.

따라서 이번 학회를 "Scaling에서 완전히 벗어났다"거나 "Agent가 전체 연구를 지배했다"고 표현하는 것은 과도합니다. 보다 정확하게는, 기존 Scaling 패러다임 위에서 **모델이 실제로 어떻게 적응하고, 추론하고, 실패하며, 배치되는지를 이해하고 제어하려는 연구**가 한층 강화되었다고 볼 수 있습니다.

---

## 2. 현장에서 눈에 들어왔던 포스터들

포스터를 전부 깊게 살펴보기에는 시간이 제한되어 있었기 때문에, 현재 업무와 개인적인 연구 관심사에 연결되는 논문을 중심으로 확인하였습니다. 아래에서는 14편의 포스터를 논문별로 나열하기보다, 보면서 비슷한 문제의식을 느꼈던 연구끼리 묶어서 소개해 보겠습니다.

### 2.1 Reasoning Model을 어떻게 적응시키고, 얼마나 생각하게 할 것인가

#### Leveraging Instruction Tuning and Merging for Reasoning Model Adaptation

**[Paper (OpenReview)](https://openreview.net/pdf?id=Ja67cQ7t0G)**

![Poster: Leveraging Instruction Tuning and Merging](/assets/images/icml-2026-workshop/poster-page07.jpg)

> **한 줄 요약**: Reasoning Model에 일반 SFT를 적용한 뒤 원래 모델과 다시 병합하면, 도메인 성능과 추론 능력을 함께 가져갈 수 있다는 연구입니다.

Reasoning Model은 수학이나 코드처럼 정답을 검증할 수 있는 분야에서는 RL을 통해 성능을 높이기 쉽지만, 요약처럼 자동 검증이 어려운 분야에서는 적응이 쉽지 않습니다.

이 연구에서는 먼저 일반적인 Instruction Tuning을 수행하고, 그 결과 모델을 기존 Reasoning Model과 Weight Interpolation 방식으로 병합하였습니다. 단순 SFT로 약해졌던 Reasoning Behavior를 병합을 통해 복구하면서, Rust Coding과 Text Summarization의 과제 성능도 유지하거나 개선하였습니다.

가장 흥미롭게 본 부분은 **새로운 Reasoning Trace를 대규모로 생성하지 않고도 기존 Instruction Data를 활용할 수 있다**는 점이었습니다. 도메인 SFT 이후 추론 성능이 줄어드는 경우에 비교적 간단하게 검증해 볼 수 있는 방법으로 보였습니다.

---

#### Think in English, Answer in Korean: Efficient Adaptation of Multilingual Tool-Using Agents

**[Paper (arXiv)](https://arxiv.org/pdf/2606.31648)**

![Poster: Think in English, Answer in Korean](/assets/images/icml-2026-workshop/poster-page09.jpg)

> **한 줄 요약**: 한국어 요청을 이해하되 내부 Tool Reasoning은 영어로 수행하고, 최종 답변은 다시 한국어로 제공하는 Enterprise Agent 학습 사례입니다.

Cohere와 LG CNS가 공동으로 개발한 **LuckyStar 111B**에 대한 연구입니다. 이미 Post-Training된 Command A를 기반으로 Multilingual SFT, 수학·NL2SQL·Tool Use RLVR, DPO, Language-Consistency Reward를 적용하였습니다.

한국어 사용자 요청을 받더라도 Schema, SQL, Tool Documentation은 영어로 구성되는 경우가 많습니다. 이 연구는 이러한 현실적인 환경을 전제로, **영어로 Reasoning과 Tool Use를 수행한 뒤 최종 사용자 응답은 한국어로 생성**하도록 학습하였습니다. 또한 4-bit Quantization을 적용하여 단일 80GB GPU에서 서빙할 수 있는 구성도 함께 제시하였습니다.

이번에 본 논문 중 **한국어 Enterprise Agent 업무 맥락에 가장 직접적으로 연결되는 연구**였습니다. 특히 Language-Consistency Reward와 검증 가능한 NL2SQL·Tool Workflow에 RLVR을 적용한 방식은 소규모로 재현해 볼 가치가 있어 보였습니다.

---

#### CURVE: Dynamic Cost-Accuracy Control for LLM Reasoning via Online Constrained Optimization

![Poster: CURVE](/assets/images/icml-2026-workshop/poster-page11.jpg)

> **한 줄 요약**: 모든 문제에 같은 추론 비용을 쓰지 않고, 문제별로 필요한 Verification 횟수를 동적으로 조절하는 연구입니다.

일반적인 Reasoning System은 쉬운 문제와 어려운 문제에 동일한 Sampling 또는 Verification Budget을 적용하는 경우가 많습니다. 이 방식은 쉬운 문제에는 계산을 낭비하고, 어려운 문제에는 충분한 자원을 제공하지 못할 수 있습니다.

CURVE는 여러 Verifier의 점수와 불확실성을 이용하여 답변을 언제 확정할지 결정하고, 목표 Risk와 Coverage 조건을 만족하도록 **Threshold를 온라인으로 조절**합니다. 포스터에서는 쉬운 문제는 빠르게 종료하고, 어려운 문제에는 더 많은 Verification을 사용하는 형태가 나타났습니다.

API 비용이 중요한 서비스에서는 Accuracy를 조금 더 높이는 것보다 **어떤 요청에 얼마만큼의 계산을 사용할지 결정하는 Controller가 더 실용적인 문제**가 될 수 있다는 생각이 들었습니다. 공개 논문 원문은 확인하지 못했습니다.

---

#### Distribution-Calibrated Inference-Time Compute for Thinking LLM-as-a-Judge

**[Paper (arXiv)](https://arxiv.org/pdf/2512.03019)**

![Poster: Distribution-Calibrated Inference-Time Compute](/assets/images/icml-2026-workshop/poster-page12.jpg)

> **한 줄 요약**: LLM Judge를 여러 번 호출했을 때 단순 다수결을 하지 않고, 승리·패배·동률의 분포를 보정하여 판정의 신뢰도를 높이는 연구입니다.

LLM-as-a-Judge를 한 번만 호출하면 결과가 불안정할 수 있기 때문에, 여러 개의 Thinking Sample을 생성하고 다수결로 판정하는 방식이 자주 사용됩니다.

하지만 평가 결과에 Tie가 포함되는 경우에는 단순 Majority Vote가 충분하지 않을 수 있습니다. Judge Model이나 Prompt에 따라 Tie를 선택하는 비율 자체가 다를 수 있기 때문입니다.

이 연구는 A 승리, B 승리, Tie의 Count를 **Bradley–Terry–Davidson 형태의 3-Way Preference Model로 보정**하여 최종 판단을 내립니다. 단순히 어느 쪽 표가 많은지만 보는 것이 아니라, 비동률 표의 차이와 전체 비동률 비율을 함께 사용합니다.

평가 파이프라인에서 LLM Judge를 여러 번 호출하고 있다면, **호출 횟수를 늘리는 것만큼 결과를 어떻게 집계할지가 중요하다**는 점을 보여주는 연구였습니다.

---

#### Does Reasoning Improve Seeing? Understanding When Vision-Language Models Benefit from Thinking

**[Paper (OpenReview)](https://openreview.net/pdf?id=ieUs1hK3HG)**

![Poster: Does Reasoning Improve Seeing?](/assets/images/icml-2026-workshop/poster-page14.jpg)

> **한 줄 요약**: VLM의 Thinking Mode는 항상 유용한 것이 아니며, 답변 생성 전 내부 표현을 보면 Reasoning이 필요한 문제인지 미리 예측할 수 있다는 연구입니다.

이 연구는 Vision-Language Model에서 Thinking Mode가 어떤 문제에 도움이 되는지를 분석합니다. 쉬운 문제에서는 긴 Reasoning이 추가 비용만 발생시키는 반면, 복잡한 시각 추론이 필요한 문제에서는 성능 향상에 도움이 될 수 있습니다.

연구진은 답변 생성 전 마지막 Prefill Token의 Representation을 이용해 문제 난이도, Reasoning의 필요 여부, 적정 Reasoning Length를 예측하였습니다. 또한 특정 Attention Head의 동작을 분석하여 계산량을 조절하는 방법도 제시하였습니다.

LLM뿐 아니라 **VLM에서도 문제별로 Thinking Budget을 다르게 주는 Adaptive Compute가 가능하다**는 점이 인상적이었습니다.

---

### 2.2 Reasoning Trace는 정말 읽을 수 있고, 실제로 숙고한 결과일까?

#### Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?

**[Paper (arXiv)](https://arxiv.org/pdf/2603.20508)**

> **한 줄 요약**: 성능이 가장 높은 Reasoning Model이 반드시 다른 모델이 이해하고 학습하기 좋은 Trace를 생성하는 것은 아니라는 연구입니다.

Reasoning Model의 품질은 보통 최종 정답률로 평가합니다. 하지만 Distillation이나 Multi-Agent 협업에서는 **Trace가 얼마나 이해하기 쉽고 다른 모델에게 도움이 되는지**도 중요합니다.

이 연구는 강한 모델의 Reasoning Prefix를 약한 Student Model에 제공했을 때 정답 가능성이 얼마나 높아지는지를 **Transfer Utility**로 측정합니다. 약 9만 개의 Trace와 12개 Reasoning Model을 분석한 결과, 정답률이 높은 모델의 Trace가 Legibility 측면에서는 낮은 순위를 기록하는 경우가 있었습니다.

이 결과는 한때 관심 있게 봤던 Reasoning Trajectory 기반 Distillation과도 직접 연결됩니다. Teacher를 고를 때 정답을 잘 맞히는가뿐 아니라 **Student가 실제로 배울 수 있는 풀이를 생성하는가**도 함께 봐야 한다는 의미이기 때문입니다.

---

#### Do Thinking Tokens Help with Safety?

**[Paper (arXiv)](https://arxiv.org/pdf/2606.25013)**

![Poster: Do Thinking Tokens Help with Safety?](/assets/images/icml-2026-workshop/poster-page17.jpg)

> **한 줄 요약**: 모델이 길게 안전성을 고민하는 것처럼 보이더라도, Refusal 여부는 Thinking이 시작되기 전에 이미 거의 결정되어 있을 수 있다는 연구입니다.

Reasoning Model은 최종 답변 전에 긴 Thinking Trace를 생성합니다. 이 과정이 모델에게 안전 원칙을 숙고할 시간을 제공할 것이라는 기대가 있지만, 본 연구의 결과는 조금 달랐습니다.

여러 Open-Weight Reasoning Model에서 **첫 번째 Token의 Hidden Representation만으로도 최종 Refusal 또는 Compliance를 높은 정확도로 예측**할 수 있었습니다. 또한 최종 판단은 Thinking 과정 초반 이후에는 거의 바뀌지 않았습니다.

즉, 텍스트에는 "잠시 생각해 보자", "안전 정책을 고려해야 한다"와 같은 표현이 나타나더라도 **실제 내부 판단은 이미 고정되어 있을 수 있습니다.**

이 논문을 보면서 긴 CoT가 곧 실제 숙고를 의미하지는 않는다는 점이 다시 인상 깊었습니다. Safety 평가에서도 Trace의 문구보다 실제 Decision Revision이 있었는지를 측정할 필요가 있어 보입니다.

이 논문은 ICML 2026의 Outstanding Paper가 아니라, **AI4GOOD Workshop Spotlight**와 **Mechanistic Interpretability Workshop Poster**로 발표된 연구입니다.

---

#### LaRA: Layer-wise Representation Analysis for Detecting Data Contamination in RL Post-Training

**[Paper (arXiv)](https://arxiv.org/pdf/2605.29888)**

![Poster: LaRA](/assets/images/icml-2026-workshop/poster-page19.jpg)

> **한 줄 요약**: RL로 학습된 모델의 데이터 오염 여부를 출력 확률이 아니라 Layer별 Representation Geometry로 탐지하는 연구입니다.

기존의 Data Contamination 탐지는 학습 데이터에 포함된 문장에서 Likelihood가 높거나 Entropy가 낮아지는 현상을 주로 사용합니다. 하지만 RL Post-Training은 Token Likelihood보다 전체 Trajectory Reward를 최적화하기 때문에 기존 지표가 불안정할 수 있습니다.

LaRA는 질문의 핵심 정보를 제거하거나 의미가 유사한 변형을 만든 뒤, 모델의 Layer별 Hidden Representation이 어떻게 변하는지를 측정합니다. 구체적으로 **Perturbation Sensitivity, Directional Collapse, Local Representation Rigidity**를 사용합니다.

실험에서는 오염된 Sample이 깊은 Layer로 갈수록 다른 Geometry를 보였고, 기존 Output-Level 방식보다 높은 탐지 성능을 기록했습니다.

RLVR 실험에서 **Benchmark 성능이 갑자기 크게 상승했을 때, 실제 Reasoning이 향상된 것인지 학습 데이터가 섞인 것인지 구분**하는 데 활용할 수 있는 방법으로 보였습니다.

---

### 2.3 Agent는 무엇을 기억하고, 여러 Agent는 어떻게 협업해야 할까?

#### PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents

**[Paper (arXiv)](https://arxiv.org/pdf/2603.03296)**

![Poster: PlugMem](/assets/images/icml-2026-workshop/poster-page21.jpg)

> **한 줄 요약**: Agent의 원시 대화 기록을 그대로 검색하지 않고, 재사용 가능한 사실과 행동 지침으로 변환하여 저장하는 범용 Memory Module입니다.

기존 Agent Memory는 전체 대화나 Trajectory를 Chunk 단위로 저장하고 검색하는 경우가 많습니다. 기록이 길어질수록 Context가 비대해지고, 실제 의사결정에 필요한 정보가 희석될 수 있습니다.

PlugMem은 경험을 다음 두 종류의 지식으로 구조화합니다.

- **Propositional Knowledge**: 사실이나 상태에 관한 지식
- **Prescriptive Knowledge**: 특정 상황에서 무엇을 해야 하는지에 관한 절차적 지식

이 지식을 **Knowledge-Centric Memory Graph**로 구성하고, 여러 종류의 Agent에 동일한 Memory Module을 적용합니다. LongMemEval, HotpotQA, WebArena처럼 성격이 다른 과제에서도 일관된 개선을 보였습니다.

Agent가 모든 것을 기억하는 것보다 **다음 의사결정에 필요한 형태로 기억을 재구성하는 것이 중요**하다는 점이 핵심으로 느껴졌습니다.

---

#### Flag Game: Interpreting Decision Mechanisms of Bounded Social Agents

**[Paper (OpenReview)](https://openreview.net/pdf?id=4uxDZTYd7U)**

![Poster: Flag Game](/assets/images/icml-2026-workshop/poster-page23.jpg)

> **한 줄 요약**: 여러 Agent에게 서로 다른 일부 정보만 제공했을 때, Agent 수와 소통 방식에 따라 집단 지성이 향상되기도 하고 양극화되기도 한다는 연구입니다.

각 Agent는 국가 국기의 일부 Crop만 관찰한 뒤 어떤 국가인지 추정합니다. 이후 Pairwise, Broadcast, Manager 등의 방식으로 정보를 교환하며 최종 집단 결정을 내립니다.

흥미로운 결과는 **Agent 수가 많아질수록 성능이 계속 높아지는 것이 아니었다**는 점입니다. Agent가 너무 적으면 정보가 부족하지만, 너무 많아지면 잘못된 의견이 강화되거나 Polarization이 발생할 수 있었습니다.

또한 동일한 모델만 여러 개 사용하는 것보다 **서로 다른 오류 패턴을 가진 모델을 혼합**했을 때 더 좋은 결과가 나타났습니다.

Multi-Agent 시스템을 설계할 때 단순히 Agent 수를 늘리기보다 서로 다른 정보와 역할을 갖게 하고, 어떤 구조로 소통하게 할지를 설계하는 것이 중요하다는 점을 보여주는 연구였습니다.

---

#### Keep It Simple: Multi-Key Episodic Memory Retrieval for Ultra-Long Video Understanding

![Poster: Keep It Simple (MERIT)](/assets/images/icml-2026-workshop/poster-page25.jpg)

> **한 줄 요약**: 수 시간 길이의 영상을 복잡한 Graph로 구성하지 않고, 여러 종류의 Key를 이용해 관련 장면을 찾아가는 Memory Retrieval 연구입니다.

Ultra-Long Video QA에서는 전체 영상을 한 번에 VLM Context에 넣을 수 없습니다. **MERIT**는 영상을 짧은 Clip으로 나누고 각 Clip에 **Event, Dialogue, Object, Summary**라는 여러 종류의 Key를 생성합니다.

질문이 들어오면 각 Key를 이용해 관련 Clip을 찾고, 시간적으로 인접한 장면을 확장한 뒤 필요한 정보만 VLM에 제공합니다.

이 연구는 Agent Memory 논문은 아니지만, **장기 기억을 구성할 때 하나의 Embedding이나 하나의 요약만으로 모든 질문을 처리하기 어렵다**는 점에서 PlugMem과 비슷한 문제의식을 갖고 있었습니다.

회의 영상이나 매장 영상처럼 길이가 긴 멀티미디어 데이터를 검색할 때 활용할 수 있는 아이디어로 보였습니다.

현장에서 소개된 포스터이지만, 공개 연구 목록상 게재처는 ECCV 2026으로 확인됩니다. 아무래도 조금 더 발전시켜서 ECCV에 낸 것 같습니다.

---

### 2.4 모델의 편향과 위험 신호를 출력 바깥에서 찾는 연구

#### Treat Bias as Noise: Training Bias-Robust LLM Reasoning via Reinforcement Learning

**[Paper (OpenReview)](https://openreview.net/pdf?id=NDGcCpCUiO)**

![Poster: Treat Bias as Noise](/assets/images/icml-2026-workshop/poster-page27.jpg)

> **한 줄 요약**: 편향된 단서를 제거하는 대신, 해당 단서가 정답 보상과 상관관계를 갖지 못하도록 RL 데이터를 구성하는 연구입니다.

LLM은 "전문가가 이렇게 말했다", "대다수가 이 답에 동의한다", "더 길고 상세하게 설명되어 있다"와 같이 정답과 직접 관련이 없는 단서에 영향을 받을 수 있습니다.

이 연구는 특정 Bias Cue가 정답을 지지하는 경우와 오답을 지지하는 경우를 균형 있게 구성합니다. 이를 통해 **모델이 Bias Cue를 따라가는 것이 Reward를 얻는 데 도움이 되지 않도록** 만듭니다.

한 종류의 Bias를 중심으로 학습했음에도 Authority, Distraction, Verbosity 등 다른 형태의 Bias에도 일반화되는 결과를 보였습니다.

검색 결과나 외부 문서를 사용하는 Agent는 정보의 출처, 문장 길이, 다수 의견에 쉽게 영향을 받을 수 있습니다. 따라서 Agent의 Robustness를 높이기 위해 **비인과적 단서와 Reward 사이의 상관관계를 끊는 학습 설계**가 필요하다는 점이 인상적이었습니다.

---

#### Beyond the Prompt: Leveraging Pre-Decoding States for Jailbreak Detection in dLLMs

**[Paper (OpenReview)](https://openreview.net/pdf?id=QVRvaVBwRh)**

![Poster: Beyond the Prompt](/assets/images/icml-2026-workshop/poster-page29.jpg)

> **한 줄 요약**: Diffusion Language Model에서는 답변을 생성하기 전의 Masked Response Hidden State만으로도 Jailbreak 위험을 탐지할 수 있다는 연구입니다.

기존 Guardrail은 주로 사용자 Prompt를 검사하거나, 모델이 답변을 생성한 뒤 출력 내용을 검사합니다. 하지만 Diffusion Language Model은 전체 Response Position을 Masked 상태로 두고 반복적으로 복원하기 때문에, **실제 Token이 생성되기 전에도 미래 응답과 관련된 Hidden State가 존재**합니다.

**ReFuse**는 Prompt 측 Representation과 Pre-Decoding Response 측 Representation을 각각 분류한 뒤 두 결과를 결합합니다. 두 View가 서로 다른 유형의 공격을 탐지하여, 함께 사용했을 때 Jailbreak 탐지 성능이 향상되었습니다.

일반 Autoregressive Model에 바로 적용하기는 어렵지만, **Guardrail이 문자열 필터를 넘어 모델 내부의 생성 준비 상태를 감시하는 방향**으로 확장될 수 있다는 점에서 흥미로웠습니다.

---

#### Position: Prompting Intent Should Be Audited in LLM-Assisted Peer Review

**[Paper (OpenReview)](https://openreview.net/pdf?id=HnB0EAQWYf)**

![Poster: Prompting Intent](/assets/images/icml-2026-workshop/poster-page31.jpg)

> **한 줄 요약**: 학술 리뷰에서 LLM을 사용했는지 여부만 확인할 것이 아니라, 리뷰어가 LLM에 어떤 방향의 평가를 요구했는지까지 감사해야 한다는 Position Paper입니다.

최근에는 논문 리뷰를 작성할 때 LLM을 보조 도구로 사용하는 사례가 늘고 있습니다. 하지만 단순히 "LLM을 사용했는가?"만 확인해서는 리뷰가 어떤 방식으로 편향되었는지 충분히 파악하기 어렵습니다.

예를 들어 같은 논문을 평가하더라도 "장단점을 균형 있게 검토해 달라"는 Prompt와 "논문의 문제점을 집중적으로 찾아 달라"는 Prompt는 전혀 다른 리뷰를 만들 수 있습니다. 이 논문은 이러한 차이를 **Prompting Intent**로 정의하고, 중립적인 평가를 요청한 경우와 특정 방향으로 비판을 유도한 경우를 구분해 감사해야 한다고 주장합니다.

연구진은 합성 리뷰를 이용해 Neutral Intent와 Directional Intent를 구분하는 Detector를 구성하고, 실제 Peer-Review 데이터에서 Directional Prompting이 리뷰의 문체와 비판 구조에 어떤 패턴으로 나타나는지 분석하였습니다. 특히 방향성 있는 Prompt는 리뷰 전체를 완전히 바꾸기보다, **어떤 약점을 선택하고 얼마나 강조하는지에 영향**을 줄 수 있음을 보여줍니다.

LLM을 활용한 평가나 검토 과정에서는 단순한 사용 여부보다 **어떤 목표와 방향성을 Prompt에 부여했는지 기록하고 감사할 수 있어야 한다**는 생각이 들었습니다. 이는 Peer Review뿐 아니라 문서 검토, 모델 평가, 의사결정 보조 시스템에도 연결되는 문제입니다.

---

### 2.5 학습과 생성 구조 자체를 바꾸려는 연구

#### AMUSE: Anytime Muon with Stable Gradient Evaluation

**[Paper (arXiv)](https://arxiv.org/pdf/2605.22432)**

![Poster: AMUSE](/assets/images/icml-2026-workshop/poster-page34.jpg)

> **한 줄 요약**: Muon의 빠른 학습 특성은 유지하면서, 학습 후반에 발생하는 진동을 줄이기 위해 Gradient를 계산하는 위치를 점진적으로 안정적인 평균 Trajectory 쪽으로 옮기는 Optimizer입니다.

이 논문을 이해하려면 먼저 저자가 설명한 **River-Valley Loss Landscape**를 이해할 필요가 있습니다.

신경망의 Loss Landscape에는 소수의 곡률이 매우 큰 방향과 다수의 완만한 방향이 함께 존재할 수 있습니다. 이를 강에 비유하면 다음과 같습니다.

- 곡률이 큰 방향은 강의 가파른 양쪽 벽
- 곡률이 작은 방향은 앞으로 길게 이어지는 완만한 강바닥

효율적으로 학습하려면 강의 양쪽 벽을 계속 오가는 대신, 강바닥을 따라 빠르게 앞으로 이동해야 합니다. 그러나 곡률이 큰 방향에서는 작은 이동도 Gradient를 크게 바꾸기 때문에 Optimizer가 계곡의 왼쪽과 오른쪽을 반복해서 넘나들며 진동할 수 있습니다.

Muon은 Matrix Parameter의 Momentum을 Orthogonalization하여, 일부 큰 성분만 Update를 지배하지 않도록 합니다. 이 과정은 기존 Optimizer에서 상대적으로 작았던 완만한 방향의 Update를 강화하기 때문에 강바닥을 따라 빠르게 전진하는 데 도움이 됩니다.

다만 같은 이유로 곡률이 큰 방향에 남아 있던 작은 Noise도 함께 확대될 수 있습니다. 즉, 강바닥 방향의 유용한 Signal을 키우는 동시에 계곡 벽 방향의 Noise도 키울 수 있다는 것이 Muon의 장점이자 약점입니다. 이 때문에 학습 후반에 계곡 벽 사이를 오가는 진동이 나타날 수 있습니다.

AMUSE는 이를 완화하기 위해 **빠르게 움직이는 Muon Trajectory와 과거 Parameter를 평균한 안정적인 Trajectory를 함께 사용**합니다.

중요한 점은 Parameter 자체를 단순히 평균 모델로 교체하는 것이 아니라, **Gradient를 어느 위치에서 계산할지를 조절한다**는 것입니다.

- 학습 초반에는 빠른 Muon Trajectory 근처에서 Gradient를 계산하여 빠르게 적응합니다.
- 학습이 진행될수록 Gradient Evaluation Point를 Averaged Trajectory 쪽으로 이동시킵니다.
- Averaged Trajectory는 계곡 벽 방향의 진동이 평균화되어 강바닥에 더 가까운 위치이므로, 더 안정적인 Gradient를 제공합니다.

정리하면 AMUSE는 Muon이 강바닥을 따라 빠르게 나아가는 특성은 유지하면서, 계곡 벽 사이에서 발생하는 좌우 진동을 줄이려는 방법입니다.

**Anytime**이라는 이름은 전체 학습 Step을 미리 정하고 후반에 Learning Rate를 낮추는 일반적인 Decay Schedule에 크게 의존하지 않으며, 학습을 어느 시점에서 중단하더라도 비교적 안정적인 Averaged Model을 사용할 수 있다는 의미에 가깝습니다. 다만 초기 Warmup까지 전혀 사용하지 않는다는 뜻은 아니며, 핵심은 별도의 Learning Rate Decay 없이 안정성을 확보한다는 점입니다.

Vision과 LLM Pretraining 실험에서는 AdamW, Schedule-Free AdamW, Muon보다 더 좋은 Performance–Iteration Trade-Off를 보였습니다. 다만 이는 Step당 비용이 크게 줄었다는 의미라기보다, 비슷한 Step 비용으로 더 적은 Iteration 안에 좋은 성능에 도달했다는 의미로 해석하는 것이 정확합니다.

Continued Pretraining이나 소형 모델 학습에서 Muon 및 Schedule-Free 계열과 함께 비교해 볼 만한 Optimizer로 보였습니다.

---

#### Structured Masked Diffusion for Joint Multiuser Decoding

**[Paper (arXiv)](https://arxiv.org/pdf/2605.26580)**

![Poster: Structured Masked Diffusion for Joint Multiuser Decoding](/assets/images/icml-2026-workshop/poster-page36.jpg)

> **한 줄 요약**: 여러 사용자의 중첩된 무선 신호를 복원하는 문제에 Masked Diffusion과 오류정정부호의 구조적 제약을 결합한 연구입니다.

이 연구는 LLM과 직접 관련된 논문은 아니지만, **Diffusion이 이미지와 언어를 넘어 통신 분야의 구조적 복원 문제에 적용되는 사례**라는 점에서 흥미로웠습니다.

**CIDER**는 Masked-Diffusion Refinement에 다음 요소를 결합합니다.

- 동일한 메시지를 중복 복원하지 않도록 하는 Demixing
- 오류정정부호 제약을 반영하는 Parity-Aware Propagation
- 신뢰도가 낮은 Sequence만 다시 복원하는 Quality-Guided Remasking

기존 Joint Decoder와 비슷하거나 더 나은 Symbol Error Rate를 보이면서, 설정에 따라 6배에서 100배 이상의 속도 향상을 보고하였습니다.

이 연구를 통해 **Diffusion이 단순한 생성 모델이라기보다, 불완전한 상태를 반복적으로 수정하면서 구조적 제약을 만족시키는 범용 Solver**로도 확장되고 있다는 점을 확인할 수 있었습니다.

---

## 3. 마무리

이번 ICML 2026에서는 LLM과 Agent가 여전히 큰 관심을 받고 있었지만, 단순히 모델을 더 크게 만들거나 Tool을 더 많이 연결하는 것만으로는 충분하지 않다는 문제의식도 강하게 느껴졌습니다.

포스터들을 둘러보며 반복적으로 등장했던 질문은 다음과 같았습니다.

- 모델은 실제로 언제 생각하고 있는가?
- 긴 Reasoning Trace는 누구에게 유용한가?
- Agent는 무엇을 기억해야 하는가?
- 여러 Agent를 늘리면 정말 더 좋아지는가?
- 모델이 안전 장치와 평가기를 우회하고 있지는 않은가?
- 더 높은 성능을 위해 추가한 계산이 비용만 증가시키고 있지는 않은가?

결국 최근의 연구 흐름은 **모델의 Capability를 높이는 데서 끝나지 않고, 그 능력을 실제 환경에서 신뢰할 수 있고 효율적으로 사용하기 위한 방법**으로 확장되고 있다고 느꼈습니다.

---

## References

- [Announcing the ICML 2026 Workshops and Affinity Workshops](https://blog.icml.cc/2026/04/06/announcing-the-icml-2026-workshops-and-affinity-workshops/)
- [Announcing the ICML 2026 Tutorials](https://blog.icml.cc/2026/04/02/announcing-the-icml-2026-tutorials/)
- [Announcing the ICML 2026 Awards](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/)
- [Announcing the ICML 2026 Invited Talks](https://blog.icml.cc/2026/05/18/announcing-the-icml-2026-invited-talks/)
- [ICML 2026 수상 논문 소개 — PyTorchKR](https://discuss.pytorch.kr/t/icml-2026-test-of-time-10/11112)
