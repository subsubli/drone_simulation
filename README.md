# 드론 도형-추종 Offline-RL 파이프라인

한 대의 드론이 **임의의 도형(삼각형·사각형·오각형·원, 미학습 별도 일반화)** 을 pos_err(경로 상대 위치) + lookahead(진행 방향) 상태만 보고 따라 날도록, IQL(Implicit Q-Learning, offline RL)로 정책을 학습하는 파이프라인. 실데이터 대신 GAN·확산모델로 합성한 데이터로도 학습할 수 있다(6. 생성 증강).

**검증된 결과**: 5도형(미학습 별 포함) × 양방향 × 50 seeds = 500/500 완주, 코너 오차 0.11–0.16m·원 0.016–0.036m. 실데이터를 15배(1.5M→0.1M) 줄여도, 소스를 합성으로 바꿔도 동일 성능 유지. 상세 수치·전체 실험 이력은 `EXPERIMENT_LOG.md` 참고.

아래는 전체 실행 순서. 명령을 위에서 아래로 그대로 터미널에 치면 된다. `drones`(수집·평가·시각화)와 `iql`(학습) 두 conda 환경을 오가므로 각 블록 맨 앞의 `conda activate` 를 반드시 따라간다.

claude code를 사용해서 학습하는 걸 추천합니다.

---

## 0. 준비 (한 번만)

drones 환경이 없다면 gym-pybullet-drones 공식 설치 가이드를 따르세요:
https://github.com/utiasDSL/gym-pybullet-drones#installation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
# 학습용 iql 환경이 없다면:
# conda create -n iql python=3.10 -y && conda activate iql && pip install numpy scipy torch tqdm
```

아래 명령들은 전부 **절대 경로가 그대로 박혀 있어** 변수 export 없이 복붙하면 된다. 참고로 세 주요 폴더는:
- 수집·평가·시각화: `~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples`
- 학습(main.py): `~/drone_simulation/IQL-PyTorch-main`
- 생성 증강(GAN/확산모델): `~/drone_simulation/gan`, `~/drone_simulation/diffusion`

> `drones` 환경은 pybullet+torch를 같이 import하면 OpenMP 충돌(`OMP: Error #15`)이 나므로, **정책을 태우는 스크립트(평가/시각화/DAgger) 앞에는 `KMP_DUPLICATE_LIB_OK=TRUE` 를 붙인다.** 순수 학습(main.py)은 별도 `iql` 환경이라 이 문제가 없다.

> **자세 D-gain은 배포 기준 1.0을 쓴다.** `collect_shape_dataset.py`는 기본값이 이미 1.0이지만, `collect_dagger.py`·`evaluate_trained_policy.py`는 기본값이 아직 0.3이라 **항상 `--att_d_gain_scale 1.0`(또는 `--att-d-gain-scale 1.0`)을 명시**해야 한다. D=0.3에서는 강한 킥에 기체가 뒤집혀 복구 불가에 빠지는 문제가 있었고, D=1.0으로 올리면 같은 킥에서도 실제로 복구된다(EXPERIMENT_LOG.md §27).

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

pure-pursuit expert가 도형을 도는 (state, action) 궤적을 1.5M 스텝만큼 뽑는다. `--perturb_*` 는 궤적에 무작위 위치 킥을 넣어 "이탈 후 복귀" 샘플을 만든다(offline-RL이 복구를 배우려면 필요). `--direction both`(기본)는 각 도형을 **절반은 반시계(CCW), 절반은 시계(CW)** 방향으로 돈다. 교란 강도에 따라 두 가지 데이터셋을 쓴다 — **soft v2**(약한 킥, 정밀 지향)와 **hard v2**(강한 킥, 복구력 지향). 둘 다 배포 기준 D=1.0.

```bash
conda activate drones && cd ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples

# soft v2 (약한 킥: prob 0.1 / count 2 / mag 0.3)
python collect_shape_dataset.py \
  --target_steps 1500000 \
  --shapes triangle square pentagon circle \
  --att_d_gain_scale 1.0 \
  --perturb_prob 0.1 --perturb_count 2 --perturb_magnitude 0.3 \
  --direction both \
  --output_folder data_soft_v2

# hard v2 (강한 킥: prob 1.0 / count 6 / mag 1.5)
python collect_shape_dataset.py \
  --target_steps 1500000 \
  --shapes triangle square pentagon circle \
  --att_d_gain_scale 1.0 \
  --perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5 \
  --direction both \
  --output_folder data_hard_v2

# 위 명령은 per-episode CSV들을 <output_folder>/shape_dataset/ 에 쏟아낸다. 학습은 단일 파일을 받으므로 하나로 병합:
python merge_shape_dataset.py --input_folder data_soft_v2/shape_dataset --output_file data_soft_v2/merged.csv
python merge_shape_dataset.py --input_folder data_hard_v2/shape_dataset --output_file data_hard_v2/merged.csv
```
→ `data_soft_v2/merged.csv`, `data_hard_v2/merged.csv` (초기 학습 입력). 이후 단계는 이 중 하나를 예시로 쓴다(원하는 쪽으로 경로만 바꾸면 된다).

> 만약 **여러 명이 나눠 수집**(예: 3명 × 500k)했다면, 각자의 `shape_dataset/*.csv` 를 한 폴더에 모아 위 `merge_shape_dataset.py` 를 한 번만 돌리면 된다. (아래 3단계의 병합 방식과 동일 — 이유는 그 절 참고.)

CSV 컬럼: `step, tx-x ty-y tz-z(pos_err), qx qy qz qw, vx vy vz, wx wy wz, lx ly lz(lookahead), ax ay az(action=target_vel), reward, done`.

---

## 2. 초기 학습  (iql 환경, 수 분)

```bash
conda activate iql && cd ~/drone_simulation/IQL-PyTorch-main

python main.py \
  --csv-file ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples/data_soft_v2/merged.csv \
  --log-dir  ~/drone_simulation/IQL-PyTorch-main/runs \
  --n-steps 300000 --hidden-dim 256 --beta 3.0 --tau 0.85 \
  --include-lookahead --reward-clip-min -1.0 --smoothness-coef 0.05 \
  --eval-period 100000
```
→ `~/drone_simulation/IQL-PyTorch-main/runs/merged/<timestamp>/` 에 `final.pt` / `config.json` / `obs_normalization.npz`. 이 폴더 경로가 **초기 정책** `<INIT>`.

핵심 플래그:
- `--include-lookahead` : 상태에 진행 방향(lx/ly/lz) 포함. **없으면 정책이 경로에 붙은 뒤 어디로 갈지 몰라 한 자리에서 갇힌다.**
- `--reward-clip-min -1.0` : perturbation으로 커진 reward가 V/Q를 발산시키는 걸 막음.
- `--tau 0.85` `--beta 3.0` : IQL expectile·advantage temperature. 이 값이 표준 레시피.
- (참고) 네트워크·데이터를 무작정 키우면 오히려 과적합해 held-out에서 나빠진다. hidden 256 유지 권장(단, hard v2 + DAgger×1 조합은 512+LayerNorm이 더 낫다 — EXPERIMENT_LOG.md §34e/§36).

초기 정책 폴더는 방금 만든 `.../runs/merged/` 의 최신 것 — 3단계 명령이 자동으로(`ls -td ... | head -1`) 잡으므로 따로 경로를 적을 필요 없다.

---

## 3. DAgger (코너 복구 데이터) + 재병합  (drones 환경)

초기 정책은 코너에서 갇히기 쉽다. DAgger는 **정책으로 드론을 몰되(정책이 실제 방문하는 상태를 수집), 라벨은 pure-pursuit의 정답**을 기록한다. `--perturb_*` 로 코너 근처 이탈-복귀 상태를 늘린다(이게 코너 갇힘의 실질적 해결책). **DAgger 한 번이면 완주율·정밀도 둘 다 크게 개선된다** — 데이터량이나 소스보다 DAgger 여부가 결과를 결정한다(EXPERIMENT_LOG.md §36).

```bash
conda activate drones && cd ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples

KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py \
  --run-dir "$(ls -td ~/drone_simulation/IQL-PyTorch-main/runs/merged/*/ 2>/dev/null | head -1)" \
  --shapes triangle square pentagon circle \
  --seed-start 0 --n-seeds 60 \
  --slew-max-accel 2.0 \
  --att_d_gain_scale 1.0 \
  --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 \
  --direction both \
  --output_folder dagger
```
→ `dagger/shape_dataset/*.csv`.

이제 **1단계 원본 + DAgger 데이터를 합쳐 재병합**한다.
```bash
mkdir -p final/shape_dataset
cp data_soft_v2/shape_dataset/*.csv  final/shape_dataset/
cp dagger/shape_dataset/*.csv        final/shape_dataset/

python merge_shape_dataset.py \
  --input_folder final/shape_dataset \
  --output_file  final/merged.csv
```
→ `final/merged.csv` (재학습 입력).

> **왜 `data_soft_v2/merged.csv` 와 `dagger/merged.csv` 를 그냥 이어붙이지 않는가?**
> `merge_shape_dataset.py` 는 각 per-episode CSV에 순서대로 `episode_id` 를 매긴다. 이미 병합된 두 `merged.csv` 를 concat하면 **episode_id가 겹치고 헤더도 중복**돼 학습 시 에피소드 경계(next_observations 계산)가 망가진다. 그래서 항상 **per-episode CSV(`shape_dataset/*.csv`)를 한 폴더에 모아 다시 merge** 해서 episode_id를 유일하게 재부여한다.

> **앙상블-주도 DAgger (더 정밀한 방식).** `--run-dirs`(복수)로 여러 정책을 넘기면 행동 앙상블(평균)로 드론을 몰면서 데이터를 수집한다 — 배포 시 실제로 도달하는 상태 분포에 더 가까운 DAgger 데이터를 만든다. hard v2 + 6-모델 앙상블-주도 DAgger×1 조합이 프로젝트 최고 성능(고정시간 완주율 24%→100%, 코너 오차 0.6→0.13m, 원 0.69→0.036m)을 냈다 — 상세 레시피는 EXPERIMENT_LOG.md §36.

---

## 4. 재학습  (iql 환경)

```bash
conda activate iql && cd ~/drone_simulation/IQL-PyTorch-main

python main.py \
  --csv-file ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples/final/merged.csv \
  --log-dir  ~/drone_simulation/IQL-PyTorch-main/runs \
  --n-steps 300000 --hidden-dim 256 --beta 3.0 --tau 0.85 \
  --include-lookahead --reward-clip-min -1.0 --smoothness-coef 0.05 \
  --eval-period 100000
```
재학습이 끝나면 최종 정책은 `~/drone_simulation/IQL-PyTorch-main/runs/merged/` 의 최신 폴더다. (코너가 여전히 남으면 3–4단계를 그 폴더로 `--run-dir` 지정해 한 번 더 반복 = DAgger iteration. 실무적으로는 한 번으로 충분한 경우가 많다.)

> 아래 5·7단계는 그 **최신 run(= 방금 학습한 최종 정책)을 `RUN` 변수로 자동으로 잡는다**(`RUN=$(ls -td .../runs/merged/*/ 2>/dev/null | head -1)`). 특정 정책을 지정하고 싶으면 `RUN` 을 그 폴더 경로로 바꾸면 된다.

---

## 5. 평가  (drones 환경)

`eval_ensemble.py` 가 현재 표준 평가 스크립트다 — 단일 정책도, 여러 정책 앙상블도 같은 명령으로 평가하고, 완주율(traverse)·net laps·거리 오차를 한 표로 출력한다. **att-d-gain-scale은 기본값이 1.0**이라 따로 안 줘도 된다.

```bash
conda activate drones && cd ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
RUN=$(ls -td ~/drone_simulation/IQL-PyTorch-main/runs/merged/*/ 2>/dev/null | head -1)   # 최신(= 최종) 정책

KMP_DUPLICATE_LIB_OK=TRUE python eval_ensemble.py --run-dirs "$RUN" \
  --shapes triangle square pentagon circle star \
  --seeds 500 501 502 --direction both
```
출력: 도형별 `laps mean±std`(목표 3바퀴 대비 실제 전진 바퀴 수) + `trav`(net-laps ≥ 2.0 완주 비율) + `dist mean±std`(거리 오차, m) + `p90/p99/max`(꼬리).

> ⚠️ **거리 오차만 보면 안 된다.** 이건 "가장 가까운 경로점까지 거리"라, 정책이 한 자리에 갇혀 경로 옆에 붙어만 있어도 작게 나온다. **traverse/laps로 완주 여부를 먼저 확인**하고, 그다음 거리 오차로 정밀도를 본다.

전체 검증 프로토콜(논문 재현용, seeds 500–549 × 양방향 × 5도형 = 500 롤아웃)은 `--seeds 500 501 ... 549`로 확장하면 된다.

---

## 6. 생성 증강 — GAN·확산모델로 실데이터 대체  (gan / diffusion 폴더)

실측 데이터 없이 **GAN이나 확산모델로 합성한 궤적만으로도 IQL을 학습**할 수 있다. 배포 기준 D=1.0에서, 순수 GAN 합성 데이터가 실데이터보다 모든 도형에서 더 정밀했다(원 0.015 vs 0.016m, 코너 0.088–0.101 vs 0.110–0.124m — EXPERIMENT_LOG.md §28).

```bash
cd ~/drone_simulation/gan   # 또는 ~/drone_simulation/diffusion

# 생성기 학습 (soft v2 실데이터의 (s,a,s',r) 전이로 학습)
python trajectory_gan.py --csv <soft_v2/merged.csv> \
  --steps 50000 --pe-asinh 0.05 --lambda-cons 0.1 --lambda-smooth 10.0 \
  --r1-gamma 1.0 --r2-gamma 1.0 --class-cond --offpath-batch-frac 0.5 --out gen_gan_cc

# 체크포인트에서 재학습 없이 대량 샘플링 (1.5M행 = 24000 windows)
python trajectory_gan.py --load gen_gan_cc/model.pt --n-gen 24000 --gen-offpath-frac -1.0 --out gen_pool_gan
```
생성된 풀은 `build_mix.py`로 실데이터와 원하는 비율(순수 합성 포함)로 섞어 1단계의 `merged.csv` 대신 쓸 수 있다. 상세 옵션·이론적 배경은 `gan/README.md` / `diffusion/README.md` 참고.

---

## 7. 시각화  (drones 환경)

**Top-down PNG (목표 경로 vs 실제 비행, 4도형 2×2)** — 파일로 저장:
```bash
conda activate drones && cd ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
RUN=$(ls -td ~/drone_simulation/IQL-PyTorch-main/runs/merged/*/ 2>/dev/null | head -1)   # 최신(= 최종) 정책
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths.py "$RUN" 500     # -> ./policy_paths.png
```

**3D 인터랙티브 창 (마우스로 회전)** :
```bash
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths_3d.py "$RUN" 500  # 창이 뜸, 닫으면 종료
```

---

## 8. 레포에 포함된 학습된 정책 바로 실행

1~4단계(수집·학습·DAgger)를 다시 돌릴 필요 없이, **이미 학습된 정책**으로 바로 평가·시각화할 수 있다:

- **soft v2** (약한 킥 데이터 + DAgger×2, D=1.0, 500/500 완주 + 별모양 일반화): `IQL-PyTorch-main/runs/d2_merged/08-11-26_01.01.10_vxno/`
- **hard v2 + DAgger×1** (강한 킥 데이터 + 앙상블-주도 DAgger 1회, D=1.0, hidden 512+LayerNorm, 프로젝트 최고 성능): `IQL-PyTorch-main/runs/merged/08-14-26_05.58.01_xtnp/`

각 폴더에 `final.pt`(가중치) + `config.json`(설정) + `obs_normalization.npz`(정규화)가 함께 있고, 스크립트가 이 셋을 자동으로 읽으므로 **폴더 경로만** 주면 된다.

```bash
conda activate drones && cd ~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples

RUN=~/drone_simulation/IQL-PyTorch-main/runs/d2_merged/08-11-26_01.01.10_vxno

# (1) 완주율 + 거리 오차, 양방향, 미학습 별 포함
KMP_DUPLICATE_LIB_OK=TRUE python eval_ensemble.py --run-dirs "$RUN" \
  --shapes triangle square pentagon circle star --seeds 500 --direction both

# (2) 시각화
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths.py    "$RUN" 500 ccw   # top-down PNG
KMP_DUPLICATE_LIB_OK=TRUE python viz_paths_3d.py "$RUN" 500 ccw   # 3D 창
```

- **학습 CSV가 필요 없다** — 스크립트가 `shape_dataset.run()`으로 도형 경로를 즉석 생성하고 정책이 그 위에서 rollout한다. `seed`를 바꾸면(500→501…) 새로운 도형 배치가 나오므로 held-out 일반화를 여러 개로 확인할 수 있다.
- 위 5·7단계의 `RUN=$(ls -td .../runs/merged/*/ | head -1)`은 "방금 학습한 **최신**"을 자동으로 잡는 것이고, 여기선 특정 **저장된** 모델을 경로로 직접 지정하는 차이다.
- 모든 명령 앞의 `KMP_DUPLICATE_LIB_OK=TRUE`는 pybullet+torch OpenMP 충돌 회피용(빼면 `OMP: Error #15`).

---

## 핵심 설계 노트 (왜 이렇게 하는가)

- **상태 = pos_err + quaternion + vel + ang_vel + lookahead(lx/ly/lz)**. pos_err(경로까지 수직거리)만으로는 "앞으로 어디로 갈지"를 몰라 갇힌다. lookahead(앞쪽 목표점 방향)가 진행 방향을 준다. **필수.**
- **action = target_vel(3D)**. 배포/평가 시 정책 출력에 `slew_max_accel 2.0` 슬루-레이트 제한을 다시 걸어야 저수준 PID가 추종 가능(안 걸면 급변 명령에 드론이 폭주). 평가/DAgger/시각화 스크립트는 이미 적용돼 있다.
- **자세 D-gain은 배포 기준 1.0.** D=0.3(저댐핑)에서는 강한 킥에 기체가 90°를 넘어 뒤집혀 복구 불가에 빠진다. D=1.0으로 올리면 같은 킥에서 실제로 복구된다 — 순항 정밀도를 코너에서 약 5% 희생하는 대신 실배포 가능한 복구력을 얻는 트레이드오프.
- **코너 갇힘은 DAgger + 코너 킥으로 해결.** lookahead 거리를 늘리는 것(0.3→0.5)은 실패 지점을 옮길 뿐 순개선이 아니었다 → 0.3 유지.
- **완주율을 결정하는 건 DAgger이지 원본 데이터량·소스가 아니다.** 실데이터를 15배 줄여도(1.5M→0.1M), 소스를 GAN·확산모델 합성으로 완전히 바꿔도 DAgger가 있으면 500/500 완주가 유지된다.
- **평가는 progress(net laps/traverse) 지표로.** mean|pos_err|는 갇힘을 못 잡는 함정 — 제자리 지그재그도 오차가 작게 나온다.
- 서로 다른 max_speed/max_accel 로 모은 데이터를 섞으려면 그 값을 상태에 넣어야 함(안 그러면 non-Markovian). 지금은 단일 config(2.0/2.0).

## 남은 일 (실기 배포 전)
- 코너 정밀도 다듬기(완주는 하지만 코너에서 약간 오버슈트) — pure-pursuit 자체의 코너 감속/lookahead 로직을 손봐야 하는 문제라 expert 재수집이 필요한 큰 작업. 현재 정밀도(코너 0.11–0.16m)로도 충분하다고 판단해 보류 중.
- 실제 ~2kg 드론의 max_speed/max_accel과 저수준 velocity 컨트롤러 매칭 (자세 D-gain 문제는 D=1.0으로 해결됨).
