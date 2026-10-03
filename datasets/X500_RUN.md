# X500 데이터로 IQL 실행

이 데이터는 루트 README의 도형 추종 CSV와 다르다. 관측은 25차원,
행동은 4차원 CTBR(추력, roll/pitch/yaw rate)이다. 기존 도형 모델의
3차원 target velocity 평가기와 호환되지 않는다.

저장소 루트에서 실행:

```bash
cd IQL-PyTorch-main
~/miniconda3/envs/iql/bin/python train_x500.py \
  --csv-file ../datasets/data_track.csv.gz \
  --log-dir runs_x500 --n-steps 300000 --eval-period 10000
```

README의 hidden 256, beta 3.0, tau 0.85, smoothness 0.05를 사용한다.
X500의 추락 벌점을 보존하기 위해 보상 클리핑은 하지 않는다.
seed 0으로 비행 단위 분리(학습 77,427개, 검증 22,573개)하며,
정규화 통계는 학습 데이터만으로 계산한다. CSV의 next_obs를 그대로
사용하고 terminal만 부트스트랩을 중단한다. timeout은 중단하지 않는다.

각 실행 폴더에는 final.pt, config.json, obs_normalization.npz,
dataset_split.json, progress.csv, evaluation.json을 저장한다.
검증 action MSE는 기록된 행동과의 차이이며, 비행 완주율이나 정책 수익을
측정한 값이 아니다. 비행 평가는 호환되는 X500 환경이 별도로 필요하다.

2026-10-03 실행 완료: `IQL-PyTorch-main/runs_x500/data_track/10-03-26_14.53.22_vtkg/`.
30만 스텝, CPU 단일 스레드, 학습 약 12분 27초.
최종 검증 행동 MSE 0.0181848 (평균 행동 기준값 0.0191024),
학습 행동 MSE 0.0101240. 1만 스텝의 검증 MSE 0.0144415보다 최종값이
높으므로 추가 학습이 행동 적합도를 계속 개선한 것은 아니다.
온라인 비행 성능은 측정하지 않았다.
ONNX 검증은 22,573개 held-out 관측에서 통과했고,
PyTorch 대비 최대 절대 출력 오차는 `5.07e-7`이었다.

## ONNX

onnx와 onnxruntime이 설치된 학습 환경에서:

```bash
python export_x500_onnx.py <실행_폴더> --csv-file ../datasets/data_track.csv.gz
```

이 작업 환경에서는 의존성을 저장소의 `.onnx-deps/`에 설치했다.
`PYTHONPATH=../.onnx-deps ~/miniconda3/envs/iql/bin/python`으로 실행하면 된다.

- `policy.onnx`: 입력 `observations`, float32 `[batch,25]`, **원시 관측**.
- 관측 정규화가 모델 안에 포함되므로 입력을 별도로 정규화하지 않는다.
- 출력 `actions`, float32 `[batch,4]`, 결정론적 평균 행동, 범위 `[-1,1]`.
- 추력 N = `(a0+1)/2*34.19`, 각속도 deg/s = `a[1:]*[220,220,200]`.
- 좌표계는 world NWU / body FLU. PX4 연결 시 DATASET.md의 좌표 변환을 따른다.
- `onnx_validation.json`: 검증 비행 전체에서 PyTorch와 ONNX 출력 비교 결과.

## 기존 lookahead의 의미

`shape_dataset.py`의 `PurePursuitTracker.step()`과 `policy_infer.py`는
현재 드론 위치에 가장 가까운 **계획 경로점**을 찾은 다음, 경로 순서대로
약 0.3m 앞선 점을 선택한다. 그 점에서 현재 드론 위치를 뺀 3차원 벡터가
lookahead다. 0.3초 뒤 목표점을 속도로 외삽하는 방식과는 다르다.

동일 계산을 시뮬레이터 의존성 없이 사용할 수 있다:

```python
from src.x500_lookahead import PathLookahead

# path: 진행 순서대로 배열한 거의 등간격의 폐곡선 계획 경로 (N,3)
tracker = PathLookahead(path, lookahead_dist=0.3)
nearest_error, lookahead, closest_index = tracker.compute(drone_position)
```

기존 구현과 동일하게 평균 경로점 간격으로 앞설 인덱스 수를 정한다.
반대 방향은 `path[::-1]`을 사용한다. 불균일한 경로점 간격에서는 실제
미리보기 거리가 달라지며, 8자 교차점에서는 최근접점 선택이 모호하므로
진행 상태를 고려한 추가 처리가 필요하다.

X500 관측의 위치 오차는 **그 시각 목표점** 기준이다. 기존 코드의
**최근접 경로점** 기준 오차와 의미가 다르므로 기존 25차원 모델의
입력을 임의로 바꾸면 안 된다. lookahead를 추가하려면 구간별 계획 경로와
진행 방향을 확보해 현재/다음 상태 모두에 일관되게 계산한 후 재학습해야 한다.
현재 학습 및 ONNX에는 lookahead를 추가하지 않는다.

CSV에 target 위치·속도 기록은 있지만 원래의 전체 계획 경로 및 생성
파라미터는 별도 파일로 제공되지 않았다. 기록된 target 궤적을 이용한 복원은
가능성을 따로 검증해야 하며, 구간 전환이나 추락으로 잘린 경로를 임의로
폐곡선으로 연결해서는 안 된다.
