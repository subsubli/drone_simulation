# X500 궤적 추종 Offline RL 데이터셋

PX4 SITL + Gazebo(`gz_x500`)에서 수집한 쿼드로터 궤적 추종 데이터.
transition 10만 개, 연속 비행 37회, 100 Hz.

| 파일 | 크기 | 설명 |
|---|---|---|
| `data_track.csv.gz` | 14 MB | 한 행 = 한 transition, 75개 컬럼 |
| `data_track.npz` | 26 MB | 같은 내용의 numpy 배열 |
| `data_track_stats.json` | – | 통계와 정의 |
| `shard_*.npz` | 21 MB | 원본 수집 샤드 (보통 쓸 일 없음) |

## 1. 읽기

```python
import pandas as pd
df = pd.read_csv("data_track.csv.gz")          # gzip은 pandas가 알아서 푼다

import numpy as np
d = np.load("data_track.npz")
d["observations"]        # (100000, 25) float32
d["actions"]             # (100000, 4)
d["rewards"]             # (100000,)
d["next_observations"]   # (100000, 25)
d["terminals"], d["timeouts"]
```

## 2. 컬럼

| 컬럼 | 개수 | 내용 |
|---|---|---|
| `obs_0` … `obs_24` | 25 | 정책 입력 (아래 3절) |
| `act_0` … `act_3` | 4 | 행동 (아래 4절) |
| `reward` | 1 | 아래 5절 |
| `next_obs_0` … `next_obs_24` | 25 | 다음 스텝 관측 |
| `terminal` | 1 | **추락**으로 끝난 스텝 (25개) |
| `timeout` | 1 | **스텝 예산**으로 잘린 스텝 (12개) |
| `flight` | 1 | 연속 비행 id (0–36) |
| `t` | 1 | 비행 시작부터의 경과 시간 [s] |
| `segment` | 1 | 구간 id (약 10 s마다 궤적·행동 정책이 바뀜) |
| `pos_x/y/z` | 3 | 절대 위치 [m] |
| `vel_x/y/z` | 3 | 속도 (world) [m/s] |
| `rate_x/y/z` | 3 | 각속도 (body) [rad/s] |
| `target_x/y/z` | 3 | 기준 궤적 위치 [m] |
| `target_vel_x/y/z` | 3 | 기준 궤적 속도 [m/s] |

`pos` 등 원시 값은 관측에 없는 정보를 담고 있다. **보상이나 관측을 다시 정의하고 싶으면
이 컬럼들로 전부 재구성할 수 있다.**

## 3. 관측 (25차원)

| 인덱스 | 내용 | 좌표계 |
|---|---|---|
| `[0:3]` | 위치 오차 (target − pos) | world |
| `[3:12]` | 자세 회전행렬 R, row-major | body → world |
| `[12:15]` | 선속도 | world |
| `[15:18]` | 각속도 | body |
| `[18:22]` | 직전 action 4개 | – |
| `[22:25]` | **기준 궤적 속도** | world |

- **절대 위치는 관측에 없다.** 목표까지의 오차만 들어간다. 정책이 특정 좌표를 외울 수 없고,
  평행이동에 대해 불변하다.
- 쿼터니언 대신 회전행렬을 쓰는 이유는 `q`와 `−q`가 같은 자세를 나타내는 부호 모호성 때문이다.
- `[22:25]`가 이 데이터셋의 핵심이다. 정지 목표만 있는 데이터(웨이포인트 버전)에서는 이 3개가
  항상 0이라 추종 정책을 학습할 수 없다. 여기서는 98% 구간에서 0이 아니고 평균 0.79 m/s다.
- 좌표계는 world NWU (x 앞, y 왼쪽, z 위), body FLU. PX4는 NED/FRD를 쓰므로 PX4 쪽으로
  내보낼 때는 `diag(1, −1, −1)` 변환이 필요하다.

## 4. 행동 (4차원, CTBR)

PX4 offboard의 `SET_ATTITUDE_TARGET`(자세 무시, body rate 사용)과 같은 형식이다.

| 인덱스 | 내용 | 변환 |
|---|---|---|
| `act_0` | 총 추력 | `(a+1)/2 × 34.19 N`, 호버 ≈ **0.185** |
| `act_1` | roll rate | `a × 220 °/s` |
| `act_2` | pitch rate | `a × 220 °/s` |
| `act_3` | yaw rate | `a × 200 °/s` |

모두 `[-1, 1]`로 잘려 있다. 최대 각속도는 PX4 기본값(`MC_ROLLRATE_MAX` 등)과 같다.

## 5. 보상

```
r = −( 1.0·|pos_err|² + 0.05·|vel_err|² + 0.02·|rates|² + 0.5·|Δaction|² ) − 10·crash
```

- `vel_err = target_vel − vel` (속도 추종 오차)
- `Δaction`은 직전 action과의 차이 (모터 명령 떨림 억제)
- 스텝 평균 −0.189, 비행당 평균 수익 −511

보상을 바꾸고 싶으면 원시 컬럼으로 다시 계산하면 된다.

## 6. terminal과 timeout (중요)

| | 개수 | 의미 | 학습에서 |
|---|---|---|---|
| `terminal` | 25 | 추락 (z<0.15, z>6, \|x\|,\|y\|>5, 기울기>75°) | 부트스트랩 중단 |
| `timeout` | 12 | 수집 예산으로 잘림 | **부트스트랩 계속** |

`timeout`을 `terminal`처럼 다루면 가치 추정이 왜곡된다. IQL, CQL, TD3+BC 모두 마찬가지다.

## 7. 수집 조건

- **기체**: Holybro X500 V2, PX4 공식 Gazebo 모델(`gz_x500`) 파라미터 그대로.
  질량 2.0643 kg, 추력 대 중량비 1.69
- **제어 주기**: 100 Hz (PX4 offboard, MAVLink)
- **궤적**: 원 또는 8자(Lissajous), 반경 0.8–1.5 m, 주기 6–10 s, 수직 진동 0–0.3 m.
  구간(약 10 s = 한 바퀴)마다 새로 뽑아 현재 위치로 평행이동해서 이어 붙인다
- **행동 정책** (구간마다 전환):

  | 종류 | 비율 | 내용 |
  |---|---|---|
  | expert | 52% | 기하학적 PD 제어기 (kp 4.0, kd 3.5, k_att 8.0) |
  | medium | 45% | 같은 PD에 게인 ±50% 교란 + action 노이즈 σ=0.15 |
  | random | 3% | 호버 근처 OU 노이즈, 1.5–3 s 버스트, 고도 1 m 이상에서만 |

  random 구간 뒤에 expert가 복귀하므로 **교란 후 회복** 궤적이 포함된다.
- **한 번 이륙해서 계속 비행**하며 구간만 바꾼다. 추락·텔레메트리 정지·예산 도달 시
  비행을 끝내고 SITL을 재시작한다

## 8. 통계

```
transitions   100,000            flights 37 (추락 25 / 잘림 12)
비행 길이      평균 2,703 step (최소 59, 최대 6,000)
추종 오차      평균 0.203 m, p95 0.452 m, 최대 4.90 m
속도          평균 0.95 m/s, 최대 9.48 m/s
각속도         평균 0.35 rad/s, 최대 4.92 rad/s
action 포화    0.1%
```

expert와 medium이 반반이라 D4RL의 medium-expert에 가까운 구성이다.

## 9. 학습할 때 권장 사항

- **관측 정규화**: 차원별 표준편차가 0.049–0.757로 차이가 크다. `data_track.npz`의
  `obs_mean`, `obs_std`를 쓰거나 직접 계산한다.
- **분할은 비행 단위로**: transition 단위로 섞어서 나누면 같은 비행이 train과 val에
  모두 들어간다. `flight` 컬럼으로 나눈다 (`offline_data.split_by_flight`, 82k/18k).
- **terminal ≠ timeout** (6절).
- **평가**: 이 데이터로 학습한 정책은 PyBullet 환경(`x500_env`, 관측 레이아웃 동일)이나
  Gazebo에서 온라인 평가할 수 있다. 참고로 같은 과제에서 온라인 RL(TD3)은 89.1점,
  PD 제어기는 84.4점이다.
- **행동 정책 비율이 바뀌면 결과가 달라진다.** 알고리즘 비교 실험이라면 세 알고리즘 모두
  같은 데이터셋을 써야 한다.

## 10. 다시 만들거나 늘리기

```bash
bash ta/gazebo/restart_sitl.sh
python ta/gazebo/collect.py --task track --steps 100000 --session 6000 \
    --out ta/gazebo/data_track --mix expert:0.4,medium:0.4,random:0.2
python ta/gazebo/offline_data.py --data ta/gazebo/data_track --csv
python ta/gazebo/dataset_report.py --data ta/gazebo/data_track
```

수집 속도는 약 49 step/s(실시간)다. 100만 step이면 약 5.5시간. `--task waypoint`로
바꾸면 정지 목표점 데이터가 나온다(기준 속도가 0이 되므로 추종 학습에는 쓸 수 없다).

## 11. 알려진 한계

- **관측에 노이즈가 이미 들어 있다.** PX4 EKF 추정값이라 정답 상태가 아니다.
  시뮬레이터 정답 상태를 쓰는 PyBullet 환경과는 이 점이 다르다.
- **random 비율이 3%로 낮다.** 구간이 짧고 고도 제한이 있어서다. 더 필요하면
  `--mix`와 `random_burst`를 조정해 재수집한다.
- **질량·관성 등 기체 파라미터는 고정**이다(도메인 랜덤화 없음).
- 추락 25건은 전부 medium·random 구간에서 발생했다. expert만 쓰면 추락 데이터가 거의 없다.
