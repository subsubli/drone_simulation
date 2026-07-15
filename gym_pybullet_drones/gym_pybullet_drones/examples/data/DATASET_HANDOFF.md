# `merged1.5M.csv` — 데이터셋 핸드오프 문서

드론이 임의의 도형(삼각형·사각형·오각형·원)을 따라 나는 (state, action) 궤적을 모은 **offline-RL 초기 학습 데이터셋**입니다. `gym-pybullet-drones` 시뮬레이터에서 pure-pursuit expert가 비행한 결과를 기록했습니다.

- **파일**: `gym_pybullet_drones/gym_pybullet_drones/examples/data/merged1.5M.csv`
- **크기**: 629 MB (Git LFS로 관리 — clone 시 `git lfs install` 후 받으면 실제 내용이 내려옵니다. 없으면 포인터 파일만 받아집니다)
- **규모**: 423 에피소드 / **1,502,261 스텝(행)**
- **제어 주파수**: 100 Hz (1 스텝 = 0.01초)

> ⚠️ 이건 **DAgger가 섞이지 않은 순수 expert 초기 데이터**입니다. 정책이 방문한 상태를 재라벨링하는 DAgger 데이터는 여기 없으며, 필요하면 이 데이터로 초기 정책을 학습한 뒤 별도로 수집합니다(레포의 `README.md` 참고).

---

## 1. 어떻게 만들었나

```bash
python collect_shape_dataset.py \
  --target_steps 1500000 \
  --shapes triangle square pentagon circle \
  --att_d_gain_scale 0.3 \
  --perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5 \
  --direction both \
  --output_folder data
# per-episode CSV들을 하나로 병합:
python merge_shape_dataset.py --input_folder data/shape_dataset --output_file data/merged1.5M.csv
```

- **Expert**: pure-pursuit tracker + DSLPID 속도 제어. 위치 목표 항을 끄고 오직 `target_vel`로만 제어하므로, 기록된 action이 상태 전이를 인과적으로 설명합니다.
- **양방향(`--direction both`)**: 각 도형을 절반은 반시계(CCW), 절반은 시계(CW)로 비행 → **CCW 212 / CW 211** 에피소드.
- **Perturbation(`--perturb_*`)**: 에피소드마다 6번씩 무작위 위치 킥(최대 1.5m)을 넣어 "경로 이탈 → 복귀" 샘플을 만듭니다. 이 때문에 off-path 비율이 높습니다(아래 통계).
- **`--att_d_gain_scale 0.3`**: 이 속도-전용 제어 모드에서 자세 D-gain을 낮춰 roll/pitch 진동을 줄인 설정.
- **노이즈 없음**: 이 데이터셋에는 관측 노이즈를 넣지 않았습니다(σ=0).

도형별 에피소드: triangle 106 / square 106 / pentagon 106 / circle 105.

---

## 2. 컬럼 스키마 (23개)

| 컬럼 | 차원 | 단위 | 의미 |
|---|---|---|---|
| `episode_id` | 1 | — | 에피소드 번호 (병합 시 유일하게 재부여됨. 에피소드 경계 = next_state 계산 기준) |
| `step` | 1 | 스텝 (×0.01 s) | 에피소드 내 스텝 인덱스 (100 Hz이므로 초 = step/100) |
| `tx-x, ty-y, tz-z` | 3 | **m** | **위치 오차** `target_pos − drone_pos` (절대 위치가 아님, world frame) |
| `qx, qy, qz, qw` | 4 | 무단위 (단위 쿼터니언) | 자세 쿼터니언 (PyBullet native) |
| `vx, vy, vz` | 3 | **m/s** | 선속도 (world frame) |
| `wx, wy, wz` | 3 | **rad/s** | 각속도 |
| `lx, ly, lz` | 3 | **m** | **look-ahead 벡터** = 드론 → 경로상 앞쪽 목표점 (진행 방향 신호, world frame) |
| `ax, ay, az` | 3 | **m/s** | **action = target velocity**. 학습이 예측할 대상 |
| `reward` | 1 | **m** (음수) | `−|pos_err|` (가장 가까운 경로점까지의 거리) |
| `done` | 1 | bool | 에피소드 마지막 스텝에서만 True |

**학습용 state = 16차원** = `[tx-x..tz-z (3), qx..qw (4), vx..vz (3), wx..wz (3), lx..lz (3)]`.
**action = 3차원** = `[ax, ay, az]` (target velocity, yaw rate는 항상 0이라 제외).

### 상태 설계에서 꼭 알아야 할 점
- **`tx-x/ty-y/tz-z`는 위치 오차(상대량)**입니다. 절대 위치가 아니라 "목표까지 얼마나 떨어졌나"라서, 학습된 정책이 특정 좌표가 아닌 **임의 경로**에 일반화될 수 있게 하는 핵심입니다.
- **`lx/ly/lz`(look-ahead)는 진행 방향 신호**입니다. pos_err는 "경로에서 수직으로 얼마나 벗어났나"만 알려줄 뿐 "경로 위에서 어느 쪽이 앞인가"를 못 알려줍니다. look-ahead가 그걸 알려주며, **이 컬럼을 빼면 정책이 경로에 붙은 뒤 진행 방향을 몰라 제자리에서 갇힙니다**(실측: net laps +2.9 → −0.1). 학습 시 반드시 state에 포함하세요.

---

## 3. 통계

**요약 (앞 30만 스텝 샘플 기준)**
- `|pos_err|`: median ≈ **3.1 m**, mean ≈ 5.6 m — perturbation 때문에 값이 큽니다.
- **off-path 비율(`|pos_err| > 0.2m`) ≈ 79%** — 데이터의 대부분이 "이탈 후 복귀" 상태입니다. 이건 의도된 것으로, offline-RL이 복구를 배우게 합니다(이 때문에 데이터가 순수 near-expert가 아니라 mixed-quality가 되어 value 학습이 의미 있어집니다).
- 기본 속도 한계: max_speed 2.0 m/s, max_accel 2.0 m/s²; look-ahead 거리 0.3 m.

**컬럼별 최솟값 / 최댓값 (전체 1,502,261 스텝)**

| 채널 | 단위 | min | max |
|---|---|---|---|
| `tx-x` | m | −36.585 | +27.314 |
| `ty-y` | m | −24.537 | +34.257 |
| `tz-z` | m | −33.261 | +5.217 |
| `qx` | — | −0.866 | +1.000 |
| `qy` | — | −0.866 | +1.000 |
| `qz` | — | −0.865 | +1.000 |
| `qw` | — | −0.500 | +1.000 |
| `vx` | m/s | −34.376 | +19.117 |
| `vy` | m/s | −28.276 | +15.962 |
| `vz` | m/s | −15.478 | +93.839 |
| `wx` | rad/s | −100.000 | +100.000 |
| `wy` | rad/s | −100.000 | +100.000 |
| `wz` | rad/s | −65.995 | +71.524 |
| `lx` | m | −36.746 | +27.498 |
| `ly` | m | −24.479 | +34.422 |
| `lz` | m | −33.292 | +5.225 |
| `ax` | m/s | −1.400 | +1.400 |
| `ay` | m/s | −1.400 | +1.400 |
| `az` | m/s | −1.400 | +1.400 |
| `reward` | m | −38.085 | −0.000 |

**벡터 크기 최솟값 / 최댓값**

| 벡터 | 단위 | min | max |
|---|---|---|---|
| `|pos_err|` | m | 0.000 | 38.085 |
| `|look-ahead|` | m | 0.080 | 38.171 |
| `|velocity|` | m/s | 0.000 | 93.844 |
| `|angular velocity|` | rad/s | 0.000 | 141.724 |
| `|action|` (target vel) | m/s | 0.020 | 1.400 |

> ⚠️ **극단값 주의**: `action`은 ±1.4 m/s로 깔끔하게 bound돼 있지만(target-velocity 클립), `velocity`가 최대 ~94 m/s, `angular velocity`가 ~142 rad/s까지 튀는 값은 **물리적 실제 비행이 아니라 perturbation 킥(위치를 순간 리셋)의 유한차분 아티팩트**입니다. 킥이 일어난 그 한 스텝에서 (이동거리)/(0.01s)로 속도가 순간적으로 폭발한 것이며, `pos_err`/`look-ahead`가 최대 ~38 m인 것도 킥 직후 크게 벗어난 복구 상태입니다. 이런 행은 off-path 복구 라벨로는 유효하지만, 속도/각속도의 극단값 자체를 "정상 비행 범위"로 해석하면 안 됩니다. 정상 추종 구간의 속도는 target velocity(≤1.4 m/s) 수준입니다.

---

## 4. 학습에 쓰는 법 (이 레포 기준)

```bash
# iql 환경
python main.py \
  --csv-file .../data/merged1.5M.csv \
  --log-dir  .../runs \
  --n-steps 300000 --hidden-dim 256 --beta 3.0 \
  --include-lookahead --reward-clip-min -1.0
```

- **`--include-lookahead` 필수** (lx/ly/lz를 state에 포함). 없으면 위에서 설명한 대로 정책이 갇힙니다.
- **`--reward-clip-min -1.0`**: perturbation으로 커진 reward가 V/Q를 발산시키는 걸 막습니다.
- 학습기는 이 CSV를 `(s, a, r, s')`로 읽습니다(`episode_id`로 에피소드 경계를 잡아 next_state를 만듦). 다른 프레임워크에서 쓸 때도 **에피소드 경계를 넘겨 next_state를 만들지 않도록** 주의하세요.

전체 파이프라인(초기 학습 → DAgger → 재학습 → 평가/시각화)은 레포 루트 `README.md`에 있습니다.

---

## 5. 한계 / 주의

- **노이즈 없음**: 관측 노이즈 강건성이 필요하면 별도로 넣어야 합니다(수집 시 `--obs_pos_noise_std`, 저장되는 state에만 적용).
- **코너 정밀도**: expert(pure-pursuit)가 sharp 코너를 look-ahead로 약간 넓게 돕니다(overshoot). BC/offline-RL은 expert를 넘지 못하므로, 코너를 더 타이트하게 하려면 데이터 생성 쪽(속도 프로파일/look-ahead)을 바꿔 재수집해야 합니다.
- **실기 배포 전제**: 시뮬레이터는 27g CF2X 기준입니다. 실제 드론에 쓰려면 (a) 자세 제어기 재튜닝(속도-전용 모드), (b) `max_speed`/`max_accel`을 실기 추력에 맞추기가 선행돼야 합니다.
