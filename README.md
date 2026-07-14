# 드론 도형-추종 Offline-RL 파이프라인

한 대의 드론이 **임의의 도형(삼각형·사각형·오각형·원)** 을 pos_err(경로 상대 위치) + lookahead(진행 방향) 상태만 보고 따라 날도록, IQL(Implicit Q-Learning, offline RL)로 정책을 학습하는 파이프라인.

**검증된 결과**: 4도형 × held-out seed 3개 = 12/12 완주(net laps 2.5–3.3), 경로 거리 오차 1–11cm.
(830k행, 코너-집중 DAgger 60 episode)

아래는 **약 1.5M 스텝 데이터셋** 기준의 전체 실행 순서. 명령을 위에서 아래로 그대로 터미널에 치면 된다. `drones`(수집·평가·시각화)와 `iql`(학습) 두 conda 환경을 오가므로 각 블록 맨 앞의 `conda activate` 를 반드시 따라간다.


claude code를 사용해서 학습하는 걸 추천합니다

현재 초기 데이터를 1.5M으로 잡고 학습 데이터를 만듭니다
만약 전체 데이터를 1.5M으로 잡고 싶다면 1번에서 1500000이란 숫자를 1000000 근방으로 바꾸고 테스트

0.

---

## 0. 준비 (한 번만)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 학습용 iql 환경이 없다면:
# conda create -n iql python=3.10 -y && conda activate iql && pip install numpy scipy torch tqdm
```

아래 명령들은 전부 **절대 경로가 그대로 박혀 있어** 변수 export 없이 복붙하면 된다. 참고로 두 주요 폴더는:
- 수집·평가·시각화: `/Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples`
- 학습(main.py): `/Users/hanjakp/drone_simulation/IQL-PyTorch-main`

> `drones` 환경은 pybullet+torch를 같이 import하면 OpenMP 충돌(`OMP: Error #15`)이 나므로, **정책을 태우는 스크립트(평가/시각화/DAgger) 앞에는 `KMP_DUPLICATE_LIB_OK=TRUE` 를 붙인다.** 순수 학습(main.py)은 별도 `iql` 환경이라 이 문제가 없다.

---

## 속도 / device — 벤치 결과 (Mac은 기본이 이미 최적)

학습(`main.py`)만 device가 의미 있다(수집·평가·시각화는 pybullet=CPU). **그런데 이 IQL는 아주 작은 MLP(hidden 256)라, 실측하면 CPU 단일 스레드가 제일 빠르다** — 작은 텐서는 멀티스레드 동기화·GPU 커널 런치 오버헤드가 연산 이득보다 커서 병렬화가 오히려 느리다:

| 설정 | 속도(it/s) |
|------|-----------|
| **CPU 1-thread (기본)** | **~431** |
| CPU all-cores | ~199 |
| MPS (Mac GPU) | ~114 |

→ **Mac은 기본값(`--device auto` = cpu, `--threads 1`)이 이미 최적. 아무것도 안 건드려도 된다.**

`main.py` 옵션:
- `--device` : `auto`(기본, **cuda>cpu**) / `cpu` / `mps` / `cuda`. **MPS는 이 규모엔 느려서 auto가 자동 선택하지 않는다** — 굳이 쓰려면 `--device mps`.
- `--threads` : CPU 스레드(기본 1). 네트워크/배치를 크게 키운 경우에만 늘려볼 것.

**CUDA GPU 머신**에서는 `--device auto` 가 cuda를 자동으로 잡는다. `iql` 환경에 CUDA torch 설치:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 드라이버 버전에 맞게
python -c "import torch; print('cuda:', torch.cuda.is_available())"     # True 여야 함
```
학습이 실제로 GPU를 쓰는지 확인 — `main.py` 가 시작할 때 device를 찍는다:
```
[INFO] device=cuda, cpu_threads=1     # <- cuda 면 GPU 사용 중. cpu 면 CUDA torch가 안 깔린 것
```
단, **지금 크기(작은 MLP)에선 CUDA도 CPU를 크게 앞서지 못한다.** 네트워크(`--hidden-dim`)나 데이터를 대폭 키운 대규모 학습에서만 GPU가 확실히 유리하고, 그때 `--batch-size 1024+` 로 처리량을 올린다.

> **이 개발 환경(Mac)에선 CUDA 실행을 테스트할 수 없다** — 그래도 안전한 이유: 학습 코드는 `torch.cuda.is_available()` / `tensor.to(device)` 같은 **표준 torch API만** 쓰고(cuda 전용 특수 코드 없음), 이 저장소의 원본이 애초에 CUDA(D4RL) 학습용이다. 실제로 Mac에서 `--device cuda` 를 강제하면 `[INFO] device=cuda` 까지 정상 출력되고 그다음 텐서를 GPU로 올리는 데서만 실패한다(= device 선택 로직은 맞고 하드웨어만 없음). 따라서 **CUDA 머신에 옮기면 위 `[INFO] device=cuda` 가 뜨고 그대로 학습**된다 — 첫 실행 때 그 줄만 확인하면 된다.

---

## 1. 데이터 수집 + 병합  (drones 환경, 약 25–30분)

pure-pursuit expert가 도형을 도는 (state, action) 궤적을 1.5M 스텝만큼 뽑는다. `--perturb_*` 는 궤적에 무작위 위치 킥을 넣어 "이탈 후 복귀" 샘플을 만든다(offline-RL이 복구를 배우려면 필요).

```bash
conda activate drones && cd /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples

python collect_shape_dataset.py \
  --target_steps 1500000 \
  --shapes triangle square pentagon circle \
  --att_d_gain_scale 0.3 \
  --perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5 \
  --output_folder data

# 위 명령은 per-episode CSV들을 data/shape_dataset/ 에 쏟아낸다. 학습은 단일 파일을 받으므로 하나로 병합:
python merge_shape_dataset.py \
  --input_folder data/shape_dataset \
  --output_file  data/merged.csv
```
→ `data/merged.csv` (초기 학습 입력).

> 만약 **여러 명이 나눠 수집**(예: 3명 × 500k)했다면, 각자의 `shape_dataset/*.csv` 를 한 폴더에 모아 위 `merge_shape_dataset.py` 를 한 번만 돌리면 된다. (아래 3단계의 병합 방식과 동일 — 이유는 그 절 참고.)

CSV 컬럼: `step, tx-x ty-y tz-z(pos_err), qx qy qz qw, vx vy vz, wx wy wz, lx ly lz(lookahead), ax ay az(action=target_vel), reward, done`.

---

## 2. 초기 학습  (iql 환경, 수 분)

```bash
conda activate iql && cd /Users/hanjakp/drone_simulation/IQL-PyTorch-main

python main.py \
  --csv-file /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples/data/merged.csv \
  --log-dir  /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs \
  --n-steps 300000 --hidden-dim 256 --beta 3.0 \
  --include-lookahead --reward-clip-min -1.0 \
  --eval-period 100000
```
→ `/Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/<timestamp>/` 에 `final.pt` / `config.json` / `obs_normalization.npz`. 이 폴더 경로가 **초기 정책** `<INIT>`.

핵심 플래그:
- `--include-lookahead` : 상태에 진행 방향(lx/ly/lz) 포함. **없으면 정책이 경로에 붙은 뒤 어디로 갈지 몰라 한 자리에서 갇힌다.**
- `--reward-clip-min -1.0` : perturbation으로 커진 reward가 V/Q를 발산시키는 걸 막음.
- (참고) 네트워크·데이터를 무작정 키우면 오히려 과적합해 held-out에서 나빠진다. hidden 256 유지 권장.

초기 정책 폴더는 방금 만든 `.../runs/merged/` 의 최신 것 — 3단계 명령이 자동으로(`ls -td ... | head -1`) 잡으므로 따로 경로를 적을 필요 없다.

---

## 3. DAgger (코너 복구 데이터) + 재병합  (drones 환경)

초기 정책은 코너에서 갇히기 쉽다. DAgger는 **정책으로 드론을 몰되(정책이 실제 방문하는 상태를 수집), 라벨은 pure-pursuit의 정답**을 기록한다. `--perturb_*` 로 코너 근처 이탈-복귀 상태를 늘린다(이게 코너 갇힘의 실질적 해결책).

```bash
conda activate drones && cd /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples

KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py \
  --run-dir "$(ls -td /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/*/ | head -1)" \
  --shapes triangle square pentagon circle \
  --seed-start 0 --n-seeds 60 \
  --slew-max-accel 2.0 \
  --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 \
  --output_folder dagger
```
> `--n-seeds` 는 초기 데이터 규모에 맞춰 잡는다: 검증은 400k 초기 데이터에 15였으니, 1.5M(약 3.75배)이면 **60** 정도(= seed당 4도형 × 60 = 240 에피소드). 더 다양한 경로/코너 상태를 커버할수록 강건해진다.
→ `dagger/shape_dataset/*.csv`.

이제 **1단계 원본 + DAgger 데이터를 합쳐 재병합**한다.
```bash
mkdir -p final/shape_dataset
cp data/shape_dataset/*.csv   final/shape_dataset/
cp dagger/shape_dataset/*.csv final/shape_dataset/

python merge_shape_dataset.py \
  --input_folder final/shape_dataset \
  --output_file  final/merged.csv
```
→ `final/merged.csv` (재학습 입력).

> **왜 `data/merged.csv` 와 `dagger/merged.csv` 를 그냥 이어붙이지 않는가?**
> `merge_shape_dataset.py` 는 각 per-episode CSV에 순서대로 `episode_id` 를 매긴다. 이미 병합된 두 `merged.csv` 를 concat하면 **episode_id가 겹치고 헤더도 중복**돼 학습 시 에피소드 경계(next_observations 계산)가 망가진다. 그래서 항상 **per-episode CSV(`shape_dataset/*.csv`)를 한 폴더에 모아 다시 merge** 해서 episode_id를 유일하게 재부여한다.

---

## 4. 재학습  (iql 환경)

```bash
conda activate iql && cd /Users/hanjakp/drone_simulation/IQL-PyTorch-main

python main.py \
  --csv-file /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples/final/merged.csv \
  --log-dir  /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs \
  --n-steps 300000 --hidden-dim 256 --beta 3.0 \
  --include-lookahead --reward-clip-min -1.0 \
  --eval-period 100000
```
재학습이 끝나면 최종 정책은 `.../runs/merged/` 의 최신 폴더다. (코너가 여전히 남으면 3–4단계를 그 폴더로 `--run-dir` 지정해 한 번 더 반복 = DAgger iteration.)

> **아래 5·6단계의 평가·시각화 명령에 박혀 있는 정책 경로**
> `/Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-15-26_00.55.01_gmjl_la_corner`
> **는 지금 이미 학습돼 있는 최종 정책(`_la_corner`)이다.** 위 1–4단계를 직접 재현해 새 정책을 만들었다면, 그 경로를 방금 만든 run으로 바꾸면 된다:
> ```bash
> ls -td /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/*/ | head -1   # 방금 만든 최신 run
> ```

---

## 5. 평가  (drones 환경) — **반드시 2개 지표를 같이 본다**

```bash
conda activate drones && cd /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
```

**(a) 경로 진행(progress) — "실제로 도형을 도는가"의 진짜 지표**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python progress_metric.py /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-15-26_00.55.01_gmjl_la_corner 500   # held-out seed (501, 502 로도 확인)
```
출력: 도형별 `coverage`(방문한 경로 비율) + `net laps`(전진 바퀴수, 목표 3). **net laps가 핵심** — 2.5~3이면 완주, 0 근처면 갇힘.

**(b) 거리 오차 — "얼마나 정밀하게 붙어 도는가"**
```bash
KMP_DUPLICATE_LIB_OK=TRUE python evaluate_trained_policy.py \
  --run-dir /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-15-26_00.55.01_gmjl_la_corner \
  --shapes triangle square pentagon circle \
  --seed 500 --slew_max_accel 2.0 \
  --output_folder /tmp/eval
```
출력: 도형별 expert vs policy 평균 tracking error(m).

> ⚠️ **거리 오차만 보면 안 된다.** 이건 "가장 가까운 경로점까지 거리"라, 정책이 한 자리에 갇혀 경로 옆에 붙어만 있어도 작게 나온다. **항상 progress 지표로 완주 여부를 먼저 확인**하고, 그다음 거리 오차로 정밀도를 본다.

---

## 6. 시각화  (drones 환경)

**Top-down PNG (목표 경로 vs 실제 비행, 4도형 2×2)** — 파일로 저장:
```bash
conda activate drones && cd /Users/hanjakp/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths.py /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-15-26_00.55.01_gmjl_la_corner 500     # -> ./policy_paths.png
```

**3D 인터랙티브 창 (마우스로 회전)** :
```bash
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths_3d.py /Users/hanjakp/drone_simulation/IQL-PyTorch-main/runs/merged/07-15-26_00.55.01_gmjl_la_corner 500  # 창이 뜸, 닫으면 종료
```

---

## 핵심 설계 노트 (왜 이렇게 하는가)

- **상태 = pos_err + quaternion + vel + ang_vel + lookahead(lx/ly/lz)**. pos_err(경로까지 수직거리)만으로는 "앞으로 어디로 갈지"를 몰라 갇힌다. lookahead(앞쪽 목표점 방향)가 진행 방향을 준다. **필수.**
- **action = target_vel(3D)**. 배포/평가 시 정책 출력에 `slew_max_accel 2.0` 슬루-레이트 제한을 다시 걸어야 저수준 PID가 추종 가능(안 걸면 급변 명령에 드론이 폭주). 평가/DAgger/시각화 스크립트는 이미 적용돼 있다.
- **코너 갇힘은 DAgger + 코너 킥으로 해결.** lookahead 거리를 늘리는 것(0.3→0.5)은 실패 지점을 옮길 뿐 순개선이 아니었다 → 0.3 유지.
- **평가는 progress 지표로.** mean|pos_err|는 갇힘을 못 잡는 함정.
- 서로 다른 max_speed/max_accel 로 모은 데이터를 섞으려면 그 값을 상태에 넣어야 함(안 그러면 non-Markovian). 지금은 단일 config(2.0/2.0).

## 남은 일 (실기 배포 전)
- 코너 정밀도 다듬기(완주는 하지만 코너에서 약간 오버슈트). pure pursuit의 로직을 조금 수정하면 결과가 바뀔 것임 현재는 pure pursuit에서 나타나는 코너 패턴이 그대로 나타남

    - compute_time_optimal_speed_profile의 n_smoothing_laps를 늘려 코너 감속을
      더 이른 지점부터 부드럽게
    - 코너 근방에서 lookahead_dist를 동적으로 조정(늘려서 더 넓게 도는 궤적 생성)
  단, 이는 expert 재수집 + 전체 재학습이 필요한 큰 작업. 현재 1~11cm 정밀도로도
  충분하다고 판단해 보류 중.

- attitude PID를 velocity-only 모드에 맞게 재튜닝, 실제 ~2kg 드론의 max_speed/max_accel·저수준 velocity 컨트롤러 매칭.
