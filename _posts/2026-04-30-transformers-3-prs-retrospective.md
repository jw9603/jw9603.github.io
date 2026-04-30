---
layout: post
title: "transformers contributor가 되어 본 3편의 PR 회고"
date: 2026-04-30
description: "Reflections on three merged PRs (#44436, #45486, #45722) over two months of contributing to transformers."
category: "Open Source"
---

## 들어가며

안녕하세요, AI Research Engineer 정지원입니다. 평소 매일 쓰는 라이브러리에 직접 기여해보면서 AI 연구자로서 한 걸음 더 나아가고 싶다는 마음으로 huggingface/transformers 오픈소스 기여를 시작했습니다. 2026년 3월부터 4월까지 약 두 달 동안 PR 3편이 머지되었는데, 그 과정을 정리해보려고 합니다.

transformers를 고른 이유는 코드 친숙도 + maintainer pool이 active하고 review가 빠르다는 점이었습니다.

---

## PR #44436 — Multimodal Continuous Batching Fix (2026 Mar, 첫 기여)

**문제**: `transformers`의 CLI 서버 (`cli/serve.py`)에서 `continuous_batching_chat_completion`이 multimodal 모델 (예: vision-language)을 받으면 `'str' object has no attribute 'to'` 에러를 내고 죽는 버그였습니다. 이미 같은 모듈의 `generate_chat_completion`에는 `get_model_modality` 분기가 있어서 multimodal 모델에서도 동작하는데, continuous batching 쪽에는 그 분기가 누락되어 있었습니다.

**Fix**: 8 line 추가. modality 체크 + non-text 모델은 일반 generate 경로로 fallback.

```python
# Continuous batching only supports text-only models
if self.get_model_modality(model, processor=processor) != Modality.LLM:
    logger.warning_once(
        "Continuous batching is not supported for non-text-only models. Falling back to regular generate."
    )
    return self.generate_chat_completion(req)
```

**회고**: 첫 PR이라 무서웠는데 maintainer가 친절하게 review해주셨고, 작은 분기 하나 추가지만 "내가 transformers 코드를 직접 고쳤다"는 사실이 인상적이었습니다.

---

## PR #45486 — Tokenizer Protobuf Decode Error Masking (2026 Apr 20, 두 번째)

**문제**: `tokenization_utils_base.py`에 다음과 같은 코드가 있었습니다.

```python
try:
    # tokenizer 생성
    ...
except import_protobuf_decode_error():
    ...
except RuntimeError:
    ...
except OSError:
    ...
```

언뜻 보면 평범하지만 함정이 있었습니다. `import_protobuf_decode_error()` 헬퍼는 **protobuf가 없을 때 `ImportError`를 raise**합니다. Python의 except-class expression은 **lazy하게 평가**되기 때문에, 진짜 try 블록 안에서 다른 예외가 떠도 except 줄을 평가하는 시점에 그 `ImportError`가 새로 raise되어 원래 예외를 **마스킹**해버립니다. 게다가 except handler 평가 자체가 raise하면서 dispatch가 종료되어 아래 `RuntimeError`, `OSError` handler도 절대 실행되지 않습니다.

protobuf는 transformers의 필수 의존성이 아니라 sentencepiece extra를 통해서만 들어오기 때문에, `pip install transformers`만 한 환경에서 tokenizer 로딩 시 발생하는 모든 예외가 "requires protobuf" 메시지로 둔갑하는 상황이었습니다.

**Fix**: 헬퍼를 protobuf가 없을 때 `()` (빈 튜플)을 반환하도록 변경. 빈 튜플은 어떤 예외와도 매칭되지 않으므로 원래 예외가 그대로 propagate됩니다.

```python
def import_protobuf_decode_error(error_message=""):
    if is_protobuf_available():
        from google.protobuf.message import DecodeError
        return DecodeError
    return ()  # 빈 튜플 → except 매칭 안 됨 → 원래 예외 그대로
```

**회고**: 이 PR을 준비하던 중 두 개의 다른 PR (#45460, #45466)이 같은 줄을 손대고 있었습니다. 한 명은 issue author 동의 없이 PR을 열었고, 다른 PR은 다른 방향의 수정이었습니다. 제 PR description에 "Not a duplicate of #45460 (closed; drops DecodeError catching) or #45466 (opened without issue-author approval)"이라고 명시했고, 결국 제 접근이 머지되었습니다. 오픈소스에서 **중복 PR 회피와 코디네이션의 중요성**을 배운 사례입니다.

---

## PR #45722 — Dead Beam-Search Dummies Cleanup (2026 Apr 30, 세 번째 — 53분 머지)

**문제**: PR 거리를 찾으려고 main을 fresh clone해서 `make check-repo`를 PyTorch-less venv에서 돌렸는데 다음과 같이 fail했습니다.

```
Exception: The following objects are in the public init, but not in the docs:
 - BeamScorer
 - ConstrainedBeamSearchScorer
 - Constraint
 - ConstraintListState
 - DisjunctiveConstraint
 - PhrasalConstraint
```

**원인 추적**: 6개 클래스를 codebase 전체에서 grep했는데, `dummy_pt_objects.py` 외에는 어디에도 정의/사용/import가 없었습니다. `MIGRATION_GUIDE_V5.md`에도 mention이 없었고요. git log를 따라가보니 PR #41223 ("[v5] remove deprecated `generate` classes (constraints and beam scorers)")이 v5 cleanup의 일환으로 실제 구현 (`generation/beam_search.py`, `generation/beam_constraints.py`)을 통째로 삭제했는데, **dummy 파일은 안 건드린 채로 6개 dummy class가 잔재로 남아있었습니다**.

**왜 PyTorch-less에서만 fail하는가**: `__init__.py`의 try/except 분기 때문이었습니다.

```python
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # PyTorch 없을 때만: dummy_pt_objects를 _import_structure에 등록
    from .utils import dummy_pt_objects
    _import_structure["utils.dummy_pt_objects"] = [
        name for name in dir(dummy_pt_objects) if not name.startswith("_")
    ]
else:
    # PyTorch 있을 때: 실제 구현만 등록 (BeamScorer 같은 사라진 객체는 제외)
    ...
```

PyTorch가 있는 CI 환경에서는 6개 dummy가 `dir(transformers)`에 나타나지 않으므로 `check_repo.py`가 통과합니다. PyTorch가 없는 환경에서만 dummy가 leak되어 docs check가 fail하는 **environment-specific drift**였습니다. CI가 못 잡는 이유가 정확히 여기 있었습니다.

**Fix**: `dummy_pt_objects.py`에서 6개 dummy class 직접 삭제 (42 line deletion).

**Process**:
1. issue #45712 등록 (분석 + suggested fix 첨부)
2. 5분 만에 maintainer @Rocketknight1이 "this looks legit to me, would you be willing to make a PR?" 답변
3. PR #45722 등록
4. 38분 후 @Rocketknight1이 "Yep, looks clean to me. Thank you!"로 APPROVE + 직접 머지

**PR open → merge까지 53분.** 가장 빠른 turnaround였습니다.

---

## 3편의 Lessons Learned

### 1. PR 거리는 어디서 오는가

세 PR 모두 다른 경로로 발견했습니다:
- #44436: 평소 사용 중 만난 버그
- #45486: Github 이슈 트래커 검색
- #45722: 정합성 검사 직접 돌려서 발견

특히 세 번째는 "make check-repo를 fresh venv에서 돌려보기"라는 단순한 시도였는데, environment-specific drift를 잡아낸 사례라 인상적이었습니다. **CI에서 못 잡는 영역에 PR 거리가 있다**는 교훈을 얻었습니다.

### 2. 정책 준수의 중요성

huggingface/transformers는 최근 `CLAUDE.md`에 agentic contribution policy를 도입했습니다. 핵심:
- 코디네이션 먼저 (issue → maintainer 승인 → PR)
- 중복 PR 회피
- AI assistance 사용 시 명시
- Human submitter가 모든 변경을 end-to-end 이해/디펜드 가능해야

처음엔 이 정책이 다소 부담스러웠지만, 실제로 따라보니 **PR 머지 확률을 높이는 합리적 가드레일**이었습니다. 특히 "issue 먼저 → maintainer GO → PR" 흐름이 reviewer의 review burden을 줄이고 PR 거절 risk를 낮춥니다.

### 3. AI 도움 받되, 본인이 디펜드 가능하게

세 PR 모두 Claude의 도움을 받아 진행했고, PR description에 그 사실을 명시했습니다. 다만 다음을 지켰습니다:
- 모든 변경 line을 본인이 이해할 수 있는 수준에서만 진행
- AI가 작성한 분석/코드를 그대로 복붙하지 않고 본인 톤으로 다듬음
- maintainer 질문에 본인이 직접 답변 가능하도록 분석 깊이 review

AI는 분석/검색/draft에는 매우 유용하지만, **본인이 defend 못 하는 코드를 PR로 올리면 안 된다**는 게 이번 시리즈의 가장 중요한 교훈입니다.

---

## 다음 step

이번 경험으로 얻은 자신감을 바탕으로 앞으로 더 많은 오픈소스 기여를 이어갈 계획입니다. transformers 외에도 한국어 LLM 평가, multilingual reasoning 같은 제 연구 도메인의 라이브러리에도 기여해보고 싶습니다.

이 글이 오픈소스 기여를 시작하려는 분들에게 도움이 되었으면 좋겠습니다.

---

## 각 PR 링크

- [PR #44436 — Fix continuous batching for multimodal models](https://github.com/huggingface/transformers/pull/44436)
- [PR #45486 — Fix protobuf decode error masking in tokenizer loading](https://github.com/huggingface/transformers/pull/45486)
- [PR #45722 — Remove dead beam-search dummies from dummy_pt_objects.py](https://github.com/huggingface/transformers/pull/45722)
