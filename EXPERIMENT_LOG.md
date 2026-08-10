# Drone Offline-RL — Development & Experiment Log

> Full development log for the offline-RL shape-tracing pipeline (gym-pybullet-drones + IQL): design decisions, diagnoses, and a dated experiment timeline with all measured numbers. See `README.md` for how to run the pipeline and the shipped policies.


Goal: collect (state, action) trajectories of a simulated quadrotor (gym-pybullet-drones, CF2X) tracing shapes (triangle, square, pentagon, circle) via [shape_dataset.py](gym_pybullet_drones/gym_pybullet_drones/examples/shape_dataset.py), to compare offline RL algorithms, then augment the dataset with diffusion/GAN-generated trajectories (target mix: 0.5M real + 0.5M generated = 1M total transitions -- may be superseded, see collaborative-data-collection note below).

State (current schema, since 2026-07-14): 13-dim `[tx-x, ty-y, tz-z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]` -- position error (target - pos, NOT absolute position), native PyBullet quaternion (no Euler round-trip), velocity, angular velocity. Action: 3-dim `[ax, ay, az]` = target velocity (yaw rate dropped, always 0). Reward = `-|pos_err|` (linear distance, not squared). CSV columns exactly: `[episode_id,] step, tx-x, ty-y, tz-z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, ax, ay, az, reward, done`. All per-episode metadata (shape, center, tilt, max_speed/max_accel) dropped from CSV to save space. This schema is consumed directly by [[iql-pytorch-integration]].

Collaborative data collection plan (as of 2026-07-14): NOT just this user's data -- 3 people each collecting ~500k steps (~1.5M total) under the new schema, to be pooled. This means data-generation-pipeline changes (e.g. adding perturbation/noise for recovery-data coverage, see below) can't be decided unilaterally -- must coordinate with the other 2 collaborators first. The 3 old 1M-step datasets listed below predate this schema and this plan.

Weekend todo (as of 2026-07-11): decide which offline RL algorithms to compare, and collect the shape datasets (triangle/square first, then pentagon/circle).

Current pipeline design: pure velocity control (PID position term always zero, target_pos=current pos) so the logged action is causally sufficient to explain state transitions -- see [[feedback-offline-rl-data-consistency]]. Drift is corrected online via a PurePursuitTracker (re-anchors target direction to actual position every step) rather than a static waypoint schedule. Randomizes shape side lengths, placement, start yaw, and plane tilt (+/-30 deg) within a 5x5x5m workspace per episode. Physics mode is `pyb_drag` (aerodynamic drag on), with max_speed/max_accel caps where accel tapers off as speed approaches max_speed (mimics diminishing thrust headroom near top speed).

Deferred ideas (not yet implemented, "for later"):
- Gaussian blur/noise on generated paths as augmentation/robustness step (raised 2026-07-11).
- Quaternion logging (raised 2026-07-14) -- DONE, now part of the current schema above.
- **Recovery-data perturbation during collection** -- IMPLEMENTED 2026-07-14 (`shape_dataset.py --perturb_prob P --perturb_magnitude M --perturb_count N`): with prob P an episode gets N random mid-episode position kicks (`p.resetBasePositionAndOrientation`, position only, ~0.75-1.5m, spread over middle 60% of episode); pure-pursuit then records a real recovery trajectory. Also `--obs_pos_noise_std S` (Gaussian noise on the LOGGED tx-x/ty-y/tz-z only, not reward/control -- GPS-noise robustness aug). BOTH default off (backward compatible). BUT the decisive-diagnosis sweep proved recovery data is NOT the blocker (escalating 0->2->112->186 kicks / up to 78% off-path never fixed rollout) -- see decisive-diagnosis paragraph below. These features are still useful for robustness but are not the fix.
- max_accel rate-limit on the policy's action output -- IMPLEMENTED and CONFIRMED as a real (partial) fix, 2026-07-14: `evaluate_trained_policy.py --slew_max_accel 2.0` re-applies the pure-pursuit slew cap to the policy output at rollout. Fixed the low-level tracking blow-up (|commanded-achieved| ~6 -> ~0.1) but not the deeper BC-accumulation drift. Keep it on at every rollout.

Real target hardware, corrected 2026-07-14 (earlier assumption of "Crazyflie-class hardware" was wrong): a single custom-built drone, ~2kg+ total weight, 22V (6S) power, two 10000mAh batteries in the pack, using drone-show-grade GPS (implies RTK-class, cm-to-decimeter precision, not consumer ±2-5m GPS) -- not a multi-drone swarm/formation. This matters a lot: the simulated CF2X is 27g, so the real drone is ~75-90x heavier -- completely different vehicle class, dynamics, and inertia. Before real flight this requires (a) real motor/prop thrust-to-weight data (not yet known) to set physically-achievable `max_speed`/`max_accel`, and (b) separately, real attitude-controller retuning (see blocker below) -- both are config/tuning tasks, not URDF-matching-precision work, given the action interface is target_vel (see next paragraph).

Why exact drone-spec matching (URDF mass/inertia/thrust coefficients) can be deferred: the trained policy's action space is target_vel (m/s), not raw motor RPM/PWM -- a meaningfully portable abstraction across different physical drones, *as long as* the real hardware's own low-level controller can reliably track velocity commands (which still must be tuned per-drone) and the top-level `max_speed`/`max_accel` roughly match what the real drone can achieve (already exposed as CLI args, cheap to adjust and re-collect).

Known pre-real-flight blocker: with the current pure-velocity control scheme (target_pos=current pos, driven only via target_vel through DSLPIDControl's D-term), simulated roll/pitch oscillate continuously at ~1-2Hz (broad peak, not a single tone; FFT-measured), a real closed-loop dynamical/ringing behavior -- confirmed NOT a simulation-precision artifact (control_freq_hz swept 100/200/500Hz, oscillation amplitude and tracking error were identical across all three, ruling out discretization/finite-difference noise as the cause). `corr(|roll|,tracking_error) ~ 0.3` (weak), so it doesn't badly hurt position tracking, but it's a genuine attitude-loop damping-margin issue that real-world disturbances (wind, IMU noise, motor variance -- none modeled in sim) could worsen. Root cause: `DSLPIDControl`'s default gains are tuned for normal position-target control (as in this repo's cf.py etc.), not this pipeline's velocity-only mode. Mitigation applied: scaling the attitude D-gain down via `--att_d_gain_scale 0.3` (new CLI flag on `shape_dataset.py`/`collect_shape_dataset.py`, scoped to that instance only -- doesn't touch `DSLPIDControl`'s shared real-hardware-validated defaults) reduced roll/pitch amplitude from ~11deg to ~9-10deg and also improved tracking error meaningfully; going much lower (tested 0.15) caused the drone to flip in sim (roll amplitude 180deg) -- the safe range is narrow, don't push further without more testing. This is NOT solvable via reward/loss shaping (the policy only controls target_vel, and even smooth target_vel commands still produced the oscillation) nor via P-gain adjustment (not pursued, judged not worth the added tuning risk for a weekend-scoped project) nor via yaw-alignment redesign (would just relocate the ringing to a different axis and break the clean zero-yaw / no-yaw-in-state pipeline design). Real fix requires proper controls-engineering work (isolated step-response testing, systematic P+D retuning, possibly filtering DSLPIDControl's finite-difference rate estimate or using the physics engine's already-accurate `cur_ang_vel` which `_dslPIDAttitudeControl` currently ignores) -- explicitly deferred to real-hardware tuning time, not attempted further in sim this session. Severity reassessed 2026-07-14: not literally flight-fatal (real quads routinely bank 20-45deg; sim never crashed above the 0.3 D-gain-scale safety margin) -- it's a "worth fixing for margin, not a guaranteed-crash blocker."

Three 1M-step datasets collected 2026-07-14, all with `--att_d_gain_scale 0.3`, NOT meant to be merged/mixed for training unless `max_speed`/`max_accel` are included as part of the training state (same hidden-variable/non-Markovian risk as varying them per-episode within one dataset -- see [[feedback-offline-rl-data-consistency]]):
- `dataset_1M/` -- max_speed=2.0m/s, max_accel=2.0m/s² (the original baseline, re-collected from scratch with the new gain scaling)
- `dataset_1M_speed4/` -- max_speed=4.0m/s, max_accel=2.0m/s²
- `dataset_1M_speed2_accel1/` -- max_speed=2.0m/s, max_accel=1.0m/s² (surprising finding: lower accel reduced roll/pitch amplitude the most of any lever tried, ~9deg->~5.3deg, but *worsened* tracking error, ~1.5cm->~3.0cm mean -- a real oscillation-vs-tracking-accuracy tradeoff, not a free win)
Each has a `COLUMN_GUIDE.txt` (full-name column explanations) and shares one `examples/DATA_GENERATION_EXPLAINED.txt` (methodology writeup) -- both meant to be sent alongside the CSVs to a collaborator.

Offline RL algorithm choice notes (2026-07-14): comparing IQL, CQL, TD3+BC as originally planned. Given this dataset is generated by a fairly consistent near-expert scripted controller (pure-pursuit + PID), not diverse/mixed-quality data, CQL's explicit conservatism (its main strength on mixed-quality D4RL-style datasets) is less needed here and risks being overly cautious; TD3+BC and IQL are expected to be a better fit for this specific dataset's character, though the plan is still to empirically compare all three, not skip any. Training compute is not expected to be a bottleneck (small MLPs, ~1M transitions fits comfortably on a MacBook Air M5 / 32GB RAM either on CPU or partial MPS acceleration).

IQL-PyTorch-main integration (2026-07-14), name-anchored as [[iql-pytorch-integration]]: adopted `IQL-PyTorch-main/` (gwthomas-style PyTorch reimplementation) over the official JAX implementation for PyTorch-familiarity reasons. Added `src/drone_dataset.py` (loads this project's CSV schema into the dataset dict `main.py` expects) and a `--csv-file` mode in `main.py` (mutually exclusive with `--env-name`/D4RL mode, lazy-imports gym/d4rl so the folder works standalone with just `numpy scipy torch tqdm`). Dedicated `iql` conda env created (`numpy scipy torch tqdm` only) to avoid an OpenMP runtime conflict (`OMP: Error #15`) that occurs when pybullet and torch are both imported in the same env/process -- workaroundable with `KMP_DUPLICATE_LIB_OK=TRUE` when a script genuinely needs both (e.g. the rollout evaluator below). Hyperparameters decided: `tau=0.8` (vs. default 0.7 -- reasoning: this task's dense per-step reward + narrow near-expert data resembles locomotion more than AntMaze, but leaning slightly higher since data quality is very consistent), `beta=3.0` (kept at locomotion-paper default, not AntMaze's 10.0, for the same dense-reward-continuous-control reasoning). Added an action-smoothness term (`smoothness_coef`, now default 0.05) directly into `src/iql.py`'s policy loss -- penalizes `||policy(obs) - policy(next_obs)||²` using the batch's existing `next_observations`, deliberately NOT implemented via reward shaping in `shape_dataset.py` (user was explicit: keep this on the IQL side, not the data-generation side). Also added, as CLI knobs on `main.py` (all default off / original behavior): `--reward-clip-min` (floors reward, e.g. -1.0 -- this DID fix a real V/Q-loss divergence that appeared once perturbation widened the reward range from ~-0.05 to ~-3; keep it on when training on perturbed data), `--pos-err-scale` (fixed normalization divisor for the pos_err channels), `--max-action` (manual tanh action-bound override), `--oversample-offpath-frac` (fraction of each batch drawn from off-path rows, using an `offpath_mask` now returned by `drone_dataset.py`). Of these only `--reward-clip-min` proved important; `--pos-err-scale` and oversampling were diagnostic and did not fix rollout (see decisive-diagnosis paragraph below).

**Major finding from rollout evaluation (2026-07-14)**: trained a real (non-smoke-test) IQL run on a fresh 100k-step/28-episode dataset (`dataset_100k_v2`, new schema, `att_d_gain_scale=0.3`), then built `evaluate_trained_policy.py` (in the gym_pybullet_drones examples dir) to roll the trained policy through the actual PyBullet sim via a new optional `policy_fn` hook added to `shape_dataset.py`'s `run()` (default `None` = fully backward compatible; when set, overrides the pure-pursuit's `target_vel` with the policy's output while reusing the same path-generation/reward bookkeeping, for a fair expert-vs-policy comparison on held-out seeds). Result: expert mean tracking error ~1-3cm; trained-policy rollout ~2.7-2.85m (order of the whole path radius -- essentially not tracking). Root cause chain, diagnosed in order:
1. First hypothesis (state normalization missing) -- added obs mean/std normalization (fit on train CSV, saved to `obs_normalization.npz`, reapplied identically at eval time). Did not fix it (error got slightly *worse*), ruling this out as the primary cause.
2. Inspecting the raw rollout CSV found the real smoking gun: the policy (`GaussianPolicy`, unbounded output, no Tanh) was emitting target-velocity commands >20 m/s once tracking error grew even slightly, versus the ~2 m/s the expert data ever showed -- a runaway feedback loop (bad action -> further off-distribution state -> worse action).
3. Fixed by adding an architectural `max_action` bound to both `GaussianPolicy`/`DeterministicPolicy` in `src/policy.py` (`mean = max_action * tanh(raw_mean / max_action)`, `max_action` = empirical max |action component| in the training CSV, saved as `action_bound` in `obs_normalization.npz`, reused identically at eval time -- default `None` preserves original unbounded D4RL behavior). This stopped the runaway (smoothness back to expert-like levels) but tracking error stayed ~2.7-3.0m.
4. Hypothesized deeper cause AT THE TIME (later SUPERSEDED -- see decisive diagnosis below): the pure-pursuit expert is self-correcting, so its trajectories have almost no tracking-error samples ("no recovery data"). This looked like the cause but was DISPROVEN the same evening by escalating recovery data and it not helping.

**DECISIVE DIAGNOSIS (2026-07-14 evening) -- read this, it overrides hypothesis 4 above.** Ran a full hypothesis-elimination sweep on a triangle-only 100k dataset (train seeds 0-30, eval held-out seed 500), with reward-clip stabilizing V/Q. Every data-side and input-side hypothesis was empirically REJECTED (all still ~2.7-3.9m rollout error):
   - coordinate-frame mismatch (user's hypothesis): REJECTED. Confirmed in code that pos_err (state), the logged target_vel (action), AND DSLPIDControl's interpretation of target_vel are ALL world-frame (DSLPIDControl.py line ~189 `vel_e = target_vel - cur_vel`, cur_vel is PyBullet world velocity). Also data-confirmed: the pos_err<->action coupling does NOT rotate with yaw (would have to, if it were a frame bug). The 3x3 corr looked "axis-scrambled" only because off-path pos_err vectors are direction-biased (few kick events), not because axes are swapped.
   - tanh-output saturation blow-up: REJECTED (raw pre-tanh output stays bounded ~0.5-2.5).
   - input normalization scale (`--pos-err-scale 0.1`, forcing pos_err channels to a small fixed divisor so they register larger): REJECTED (still 3.04m; raised input sensitivity but response direction was wrong).
   - data QUANTITY: REJECTED. Oversampling off-path rows to 50% of every batch (`--oversample-offpath-frac 0.5`): still 2.70m.
   - data DIVERSITY: REJECTED. Re-collected with `--perturb_prob 1.0 --perturb_count 6` = 186 distinct kicks, 78% of rows off-path: still 3.81m (even slightly worse).
   - **Open-loop probe = the key result**: the trained policy reproduces the DATASET's OWN actions on the DATASET's OWN observations with 0.057 error (5.7% of action magnitude) and +0.996 direction cosine even on far-off-path (>1m) rows. So TRAINING IS ESSENTIALLY PERFECT; the failure is 100% closed-loop (rollout), not learning and not data coverage.
   - **Two closed-loop mechanisms found, in order:** (a) LOW-LEVEL TRACKING: the policy's raw per-step target_vel has no slew-rate limit, unlike the pure-pursuit data (which caps |delta-v|/step at max_accel/freq); an MLP command that jumps between steps is not physically trackable, so commanded ~0.5 m/s produced achieved ~2-6 m/s and the drone flew off. FIXED (no retrain) by re-applying the same slew cap to the policy output at rollout via `evaluate_trained_policy.py --slew_max_accel 2.0` -- brought |commanded - achieved| from ~6 down to ~0.1 (expert-level). (b) But tracking STILL failed (~3.5m) with clean low-level tracking: rollout error accumulates monotonically from step ~5. **TRUE ROOT CAUSE = BC approximation-error accumulation (covariate shift)**: pure-pursuit is a precise feedback controller (re-anchors to actual position every step); the net approximates it with ~3.7% open-loop error; in closed loop that small residual weakens the self-correction and drift compounds. This is a fundamental limit of cloning a precise feedback controller with BC/offline-RL, and is EXACTLY why no amount/kind of data fixed it (approximation error never reaches 0).

**SOLVED (2026-07-14 late evening) -- [SEE RETRACTION BELOW; this conclusion is invalid].** The fix is three things stacked, none alone sufficient: (1) `--slew_max_accel 2.0` at rollout (fixes low-level tracking blow-up); (2) DAgger -- drive the drone with the current policy to visit its own off-path states, but log pure-pursuit's answer at those states as the label, add to the training set, retrain, repeat (directly attacks the covariate-shift/accumulation root cause); (3) **raw (non-slew) pure-pursuit labels** for the DAgger data -- the slew-limited label is near-zero at large error (physically you can only change velocity so fast from a standstill) which is too weak a recovery signal and made DAgger plateau at ~2m; the raw goal-velocity label ("head to the path at profile speed", |label|~1.4 at 3.7m error) is the right target, with the slew cap re-applied as a rollout post-step. Implementation: `shape_dataset.py` gained `dagger_relabel=True` (policy drives via `policy_fn`, logs `tracker.last_raw_target_vel`); `collect_dagger.py` (new) rolls a trained policy over many seeds to collect this; loop = collect -> combine ALL prior data -> retrain (`--reward-clip-min -1.0`) -> evaluate. Results on held-out triangle seed 500 (mean tracking error): saturation-lock plateau ~2.7-3.9m -> slew-label DAgger 2.0m (plateaued) -> raw-label DAgger iter1 0.36m -> iter2 0.274m (target <0.3m met; peak 0.68m; saturation-lock entirely gone, drone now recovers and keeps following). GENERALIZATION to unseen shapes with the triangle-ONLY-trained iter2 policy (seed 500): square 0.126m and circle 0.183m both generalize WELL (validates the pos_err/path-relative state design as genuinely shape-invariant), but pentagon still failed at 2.78m. FINAL STEP -- multi-shape DAgger (collect_dagger over all 4 shapes x15 seeds with the rawdagger2 policy, combine with all prior data = 447k rows / 131 eps, retrain 150k steps): ALL FOUR shapes now track to ~0.09m at held-out seed 500 (triangle 0.086 / square 0.096 / pentagon 0.090 / circle 0.086), pentagon fixed from 2.78m. This is the COMPLETE solution to the user's goal (a single policy following arbitrary shapes from just pos_err state). Best policy dir tagged `_multishape`. Keep `--slew_max_accel 2.0` on at every rollout and every DAgger collection. Remaining gap vs expert (~0.09m policy vs ~0.01-0.03m pure-pursuit) is the residual BC approximation error. PRECISION-TUNING ATTEMPTS (2026-07-14 night) both BACKFIRED via overfitting, so don't repeat them: (a) bigger net + longer (hidden 512, 300k steps) REGRESSED -- square blew up to 2.9m; overfitting hurts closed-loop robustness. (b) more DAgger data (80 extra episodes over seeds 0-19, 733k rows total) overfit to the collection-seed distribution -- held-out seed 500 improved to ~0.04m but seeds 501/502 got WORSE (multiple shapes blew up 1-2.4m). Also learned via multi-seed validation that the baseline's ~0.09m was partly seed-500 luck: on seeds 501/502 the baseline already had intermittent per-shape blowups (square@501 2.4m, pentagon@502 0.78m) while most shape/seed combos stay ~0.09-0.11m. So the real remaining weakness is INTERMITTENT held-out blowups (robustness), not average precision -- and neither net-capacity nor data-volume fixes it (both overfit). The multishape policy (447k rows, hidden 256, 150k steps) remains the best balance and is the final kept policy. Genuinely improving precision+robustness together would need fundamentally more path diversity (tens-to-hundreds of distinct training seeds/shapes) or a different algorithm, beyond this session. Keep hidden 256, don't scale net or data volume naively.

BC-vs-IQL check (2026-07-14 night, user asked "is this BC-mixed or pure IQL?"): the training is PURE IQL (V expectile + Q TD + AWR policy; no BC term added -- DAgger is just a data-collection strategy, the learner stays IQL). And it is NOT effectively-BC on this data, contrary to my initial guess. Two agreeing pieces of evidence on the multishape policy/data: (1) advantage-weight `exp(beta*adv)` has coefficient-of-variation 2.4 (advantage mean -0.21, std 0.37) -- the AWR weighting is far from uniform, so it's actually selecting actions; (2) retraining with `--beta 0` (pure unweighted BC, same data/net/steps) gives triangle 0.106 / square 0.105 / pentagon 0.113 but circle BLOWS UP to 3.23m at seed 500, whereas beta=3 IQL keeps all four ~0.09m. So IQL's value-based weighting genuinely contributes -- specifically to worst-case robustness (preventing the occasional blow-up), which plain BC does not. WHY the data isn't near-expert despite pure-pursuit labels: DAgger injects heavily off-path states (~78% of rows have |pos_err|>0.2) where V is low and the recovery action's advantage is high, turning the dataset mixed-quality. This VALIDATES the original IQL/CQL/TD3+BC comparison goal -- the data genuinely requires value learning, so the algorithm comparison is meaningful, not collapsing to BC. For real-hardware deployment this whole result still sits on top of the earlier prerequisites (attitude-gain retuning for velocity-only mode, matching max_speed/max_accel to the real 2kg drone).

**!!! RETRACTED (2026-07-14 latest) -- the "SOLVED / ~0.09m" conclusion above is WRONG, it was a METRIC ARTIFACT. !!!** The evaluation metric `mean |pos_err|` = distance to the NEAREST target-path point. That stays small even if the policy parks near one spot and zig-zags in place, because it's always close to *some* path point -- the metric never measured PROGRESS along the path. Top-down visualization (viz scripts, seed 500) showed the multishape policy completes NONE of the 4 shapes: on every shape the flown path just wanders/zig-zags near the start corner (usually the bottom edge) while the full target shape is never traced. So all the tracking-error numbers above (0.36 -> 0.274 -> 0.09m, "generalization", "IQL beats BC", precision-tuning comparisons) are comparisons of a metric that doesn't capture the actual failure, and DON'T mean the policy works. What IS still valid from the work below: slew_max_accel stops the >20m/s velocity blow-up; reward-clip fixes V/Q divergence; DAgger+raw-labels genuinely stopped the *drift-to-3.9m* saturation-lock (the policy now stays NEAR the path instead of flying off). The unresolved core problem: the policy recovers TO the path but cannot PROGRESS ALONG it -- almost certainly because the state (pos_err = perpendicular offset to nearest point + quaternion + vel + angvel) contains no explicit path-progress/tangent-direction signal, so once on-path (pos_err~0) it has no cue which way to travel and stalls. QUANTIFIED the real failure with a progress metric (patch PurePursuitTracker.step to record closest_idx during a policy rollout; coverage = unique closest indices / path_resolution; net laps = unwrapped sum of index deltas / path_resolution). Multishape policy, seed 500, target 3 laps: triangle 6.0% coverage / -0.00 laps, square 27.1% / 0.27, pentagon 19.9% / 0.21, circle 13.1% / 0.14 -- i.e. it barely leaves the start and completes essentially zero laps, while mean|pos_err| reported 0.086-0.096m "success" for the exact same rollouts. NEXT SESSION: (1) use THIS progress metric (coverage% + net laps), never trust mean|pos_err| again; (2) likely redesign the state to include a tangent/look-ahead cue (e.g. the vector to a look-ahead point, like pure-pursuit uses) so the policy knows which way to travel once on-path, then re-collect. Diagnostic scripts live in the session scratchpad (viz_full.py = top-down full-path plot, progress_metric.py = coverage/laps).

PREV-ACTION experiment (2026-07-14 latest, user asked to add a heading cue on the IQL side WITHOUT recollecting the CSV): added `--include-prev-action` to main.py / `include_prev_action` to drone_dataset.py (appends the previous step's action = raw pure-pursuit target_vel to the obs, 13->16 dims; standard last-action-in-state, no leakage) and made evaluate_trained_policy.py feed the policy its own previous raw output (keeps a separate prev_raw for the state feature and prev_slew for the slew cap). Retrained multishape, evaluated with the PROGRESS metric: coverage rose a bit (triangle 6->37%, pentagon 20->46%, square 27->19%, circle 13->30%) BUT net laps went NEGATIVE (-0.66/-0.61/-0.56/-0.06) -- the policy now wanders/reverses along the path instead of progressing forward; still never completes a lap. Conclusion: prev-action is too weak a proxy for true path-progress -- it says "which way I just went" but not "which way is forward along the path", and the policy can't recover the sign. CSV-only feature engineering is exhausted. The genuine fix is a look-ahead-point state feature (vector from the drone to a point some distance ahead on the path, in the drone/world frame -- exactly the signal pure-pursuit steers by), which needs ABSOLUTE path coords and therefore CSV RECOLLECTION (current schema stores only relative pos_err). NEXT: add a look-ahead column set to shape_dataset.py's CSV writer (e.g. lx,ly,lz = TARGET_POS[lookahead_idx] - pos), recollect, retrain with it in the state; keep evaluating with the progress metric (coverage%/net laps), never mean|pos_err|.

LOOK-AHEAD FEATURE VALIDATED (2026-07-14 latest, triangle-only proof). Implemented: `shape_dataset.py` now logs `lx,ly,lz` = `tracker.last_lookahead_vec` (drone->look-ahead point, world frame) as CSV columns, and `policy_fn` takes a 3rd arg (lookahead). `drone_dataset.py --include_lookahead` / `main.py --include-lookahead` append lx/ly/lz to the obs (+3 dims; concat order is [state13, lookahead3, prev_action3] when both on). `evaluate_trained_policy.py` + `collect_dagger.py` read `include_lookahead` from config.json and feed the tracker's live lookahead vector to the policy at rollout (make_policy_fn is now 3-arg). Pipeline (triangle only): collect diverse+perturbation with lookahead cols -> train --include-lookahead -> DAgger -> retrain. Progress-metric results (seed 500, target 3 laps), triangle net laps: 13-dim baseline 0.00 (stuck) -> prev-action -0.66 (reverses) -> lookahead initial (no DAgger) +0.58 (direction FLIPS positive) -> lookahead + 1 DAgger iter +2.60 laps / 67% coverage (near-full forward traversal!). The lookahead-trained triangle policy even generalizes to circle (+2.41 laps) though square/pentagon stay low (0.43/0.14) since only triangle was trained. CONCLUSION: the missing ingredient all along was an explicit path-progress/heading state feature; pos_err alone (perpendicular offset) can't tell the policy which way is forward. Look-ahead vector fixes it. NEXT: recollect ALL 4 shapes with lookahead cols + DAgger, retrain, expect all four to traverse. This supersedes the whole "SOLVED then RETRACTED" saga above -- lookahead is the actual fix.

User's actual goal (clarified 2026-07-14 evening): the learned policy should generalize to UNSEEN shapes/paths at deployment (pure-pursuit needs the full path in advance; the policy is meant to follow an arbitrary path from just the relative pos_err state). This is a valid goal and is what the pos_err state design targets. BUT the closed-loop-accumulation blocker must be solved first -- the policy can't even track same-family held-out paths in closed loop yet. NOTE: residual RL (`action = pure_pursuit + net_correction`) does NOT fit this goal -- its baseline (pure-pursuit) itself needs the path, so it's unavailable exactly when "the shape is unknown"; and with expert==pure-pursuit the residual target is ~0 anyway. Leading next-session directions to break the closed-loop accumulation: DAgger-style relabeling of rollout states with pure-pursuit actions (breaks offline purity but directly attacks covariate shift); maximize policy precision (bigger net / 500k+ steps to push open-loop error well below 3.7%); heavier observation-noise augmentation during collection. Keep `--slew_max_accel 2.0` on at rollout regardless (it's a real, confirmed fix for mechanism (a)).

ALL-4-SHAPES lookahead result (2026-07-15, progress-metric-verified, the REAL solution). Recollected all 4 shapes with lookahead cols (la_all_diverse ~403k rows) -> train --include-lookahead -> DAgger all 4 (la_dagger_all) -> retrain 150k (la_final, 616k rows, tagged `_la_final`). Progress metric (net laps, target 3), seed 500: triangle 2.74 / square 2.84 / pentagon 3.16 / circle 2.62 -- ALL FOUR actually traverse (baseline was 0-0.27). Top-down viz confirms the policy now traces the whole shape (vs the earlier one-corner zig-zag). Multi-seed: 10 of 12 shape-seed combos traverse (~2.4-3.2 laps); 2 intermittent failures remain (triangle@501 0.93, square@502 0.20) -- same robustness gap seen before, addressable with more DAgger/diversity, NOT a return of the core bug. Corner precision is imperfect (slight overshoot). BOTTOM LINE: the genuine fix was the look-ahead state feature (progress/heading cue) + DAgger + slew-limit + reward-clip; verified by the progress metric (coverage%/net-laps), never mean|pos_err|. Kept policy: `_la_final`; kept data: la_final/. Remaining work: tighten robustness (kill the ~2/12 intermittent stalls) and corner accuracy; then the real-hardware prereqs (attitude retune, max_speed/accel matching).

STALL ROOT CAUSE = FIRST CORNER (2026-07-15). Diagnosed the 2/12 lookahead-0.3 failures (square@502=0.20 laps, triangle@501=0.93): via a closest_idx trace, the policy flies the first EDGE perfectly (|pos_err|~0.01, smooth accel) then STALLS AT THE FIRST CORNER (square@502: closest_idx advances 0->614 by step ~300 then frozen at 614 for the remaining 3400 steps; idx 614 ~ first corner of the 750-per-edge square path). Not an early/slow-start problem (first 120 steps are perfect) and lookahead direction doesn't wobble. Seed 502 has tilt_deg=4.2 deg (vs 12-30 for neighbours) -- a nearly-flat shape whose corner is a hard 2D sharp turn. Tried raising DEFAULT_LOOKAHEAD_DIST 0.3 -> 0.5 (recollect la5_* all 4 shapes, retrain _la5_final): it FIXED square@502 (0.20->3.31) and triangle@501 (0.93->3.04) but BROKE square@500 (2.84->0.74), circle@500 (2.62->0.67), triangle@502 (2.68->0.30) -- net 10/12 -> 9/12. So lookahead LENGTH just relocates which corner stalls, not a real fix; corner traversal is intrinsically hard and under-covered in training. Keep lookahead at 0.3 (10/12 > 0.5's 9/12). Progress metric note: rollouts are time-limited to n_laps*expert_lap_time, so net laps ~3 = expert-speed full traversal; ~2.6-3.2 means near-expert-speed, not a time-limit artifact.

CORNER-FOCUSED DAGGER = the corner fix, 12/12 SOLVED (2026-07-15). Added `--perturb_prob/--perturb_count/--perturb_magnitude` to collect_dagger.py so the DAgger rollout gets perturbation KICKS during collection (policy drives + kicks knock it off near corners -> it visits corner-recovery states -> pure-pursuit labels them). Collected with the lookahead-0.3 `_la_final` policy, 4 shapes x15 seeds, perturb_prob=1.0 count=8 mag=1.5 (la_corner_dagger, 60 eps) -> combined with la_final base = 830k rows -> retrained 150k (`_la_corner`). Progress metric, all 4 shapes x seeds {500,501,502} = ALL 12 traverse, net laps 2.5-3.26, ZERO stalls (vs lookahead-0.3's 10/12 with square@502=0.20 & triangle@501=0.93 corner-stalled, and lookahead-0.5's 9/12). Net improvement, nothing broken -- unlike raising lookahead length (which just relocated stalls). BEST POLICY NOW: `_la_corner` (lookahead 0.3 + corner-focused DAgger). This is the genuine, progress-metric-verified, robust solution to "one policy traces arbitrary shapes from pos_err+lookahead state": the full recipe = lookahead state feature (lx/ly/lz) + reward-clip + slew-limit at rollout + DAgger with corner-kick perturbation. Remaining polish: corner-cutting/overshoot precision is still imperfect (coverage 43-81% while laps~2.5-3.3, i.e. it traverses but not pixel-tight); and the real-hardware prereqs (attitude retune for velocity-only mode, match max_speed/max_accel + low-level velocity controller to the real ~2kg drone) still stand before any real flight.

Corner-overshoot is an EXPERT limitation, not a policy bug (user's insight 2026-07-15): the policy charges the corner then snaps -- but that's exactly how the pure-pursuit EXPERT drives (it steers toward a look-ahead point, which cuts/overshoots at sharp corners), and BC/DAgger can only imitate the expert, never beat it. Expert's own tracking error is also larger on sharp-corner shapes (triangle/square) than on the circle for the same reason. So tightening corners CANNOT be done on the RL side -- it requires changing the DATA-GENERATION pure-pursuit corner behavior in shape_dataset.py (e.g. stronger corner deceleration in the speed profile, or an adaptive/shorter look-ahead near corners) and re-collecting. Deferred; noted as the starting point whenever corner precision is revisited.

BIDIRECTIONAL (CW+CCW) SUPPORT ADDED + SOLVED (2026-07-15, user asked "make half the data go clockwise"). Until now every episode traversed shapes counter-clockwise only (waypoints sampled at increasing angle; tracker steps toward increasing path index). Added a `clockwise` flag: `shape_dataset.run(clockwise=True)` reverses waypoint order (`TARGET_POS = TARGET_POS[::-1]`) BEFORE the speed-profile / look-ahead are computed, so the closed path is traced the other way (start point is irrelevant on a closed loop). Exposed as `--direction {both,ccw,cw}` on collect_shape_dataset.py (default `both`: toggles direction every full round through `shapes` -- NOT per-episode `idx%2`, which would pin each of the 4 shapes to one direction since period-2 toggle vs period-4 shape cycle; round-level toggle gives each shape exactly half/half), collect_dagger.py (toggle by seed parity), evaluate_trained_policy.py (`both` rolls each shape CCW and CW, labels `triangle-ccw`/`triangle-cw`), and progress_metric.py (3rd argv `cw`; net-laps stays positive since the reversed path is still traversed by increasing index). Full 1.5M-step bidirectional pipeline: collect (423 eps, 212 CCW / 211 CW, ~106 eps/shape) -> train 300k `--include-lookahead --reward-clip-min -1.0` -> DAgger-1 both dirs n-seeds 60 (240 eps) -> re-merge (663 eps, 2.36M rows) -> retrain (policy v1). v1 eval (held-out seed 500, both dirs): circle & pentagon traverse both directions (net laps 2.6-3.35, dist 1.5-12cm) BUT square FAILED both ways (net laps 0.85-1.02, dist 1.8-2.5m) and triangle-CW failed (0.85m) -- the SAME low-tilt sharp-corner stall as the single-direction saga, RE-APPEARING because doubling directions doubles the corner-recovery states to cover on the same DAgger budget. FIX = a 2nd DAgger iteration focused on the failing shapes: collect_dagger with v1, `--shapes square triangle --seed-start 60 --n-seeds 60 --direction both` (120 eps) -> re-merge init+dagger1+dagger2 (783 eps, 2.78M rows) -> retrain (policy v2). v2 eval (seed 500, both dirs): ALL 8 shape-direction combos traverse -- net laps triangle 2.71/2.69, square 2.92/2.96, pentagon 3.31/3.35, circle 2.70/2.69; distance error <=12.4cm on every combo (square went 1.8-2.5m -> 0.124m, triangle-CW 0.85m -> 0.098m). So bidirectional works with the SAME recipe as single-direction, just needing an extra corner-focused DAgger pass on the sharp-corner shapes to pay for the doubled state space. Kept data: data_all2/ (init+dagger1+dagger2). This is the current best/most-general policy (arbitrary shape, EITHER direction). Corner coverage verdict still reads `partial` (59-85%) on the angular shapes though net-laps/distance confirm full traversal -- same expert-corner-overshoot limitation as before, not a bug.

SOFT-PERTURBATION DATASET + tracker-spiral diagnosis (2026-07-15). The original 1.5M perturbation (prob 1.0 / count 6 / mag 1.5) makes off-path 79%, |pos_err| median 3.3m / max 38m -- too aggressive for a general handoff dataset. Collected a SOFT version (prob 0.1 / count 2 / mag 0.3): off-path 7%, median 0.006m / max 5.7m, i.e. 93% precise-tracking + 7% recovery. Key finding on the DATA→DAgger interaction: soft data trained INITIAL-ONLY does NOT complete (net laps 0.69-2.44, CW worst -- too few off-path samples to learn corner recovery, plus no DAgger for covariate shift). But soft + ONE DAgger pass (both dirs, corner kicks prob1.0/count8/mag1.5, n-seeds 60) = 8/8 traverse (net laps 2.67-3.31, dist ≤14cm) -- whereas the DIRTY original needed TWO DAgger passes for the same. So a CLEAN precise-tracking baseline makes DAgger more efficient (clean tracking + one corner-recovery pass beats dirty-data + 2 passes). A soft DAgger-2 pass (seed 60-119, all 4 shapes, both dirs) did NOT overfit -- multi-seed sweep (500-503) IMPROVED on DAgger-1: dist_mean 0.093→0.080m, dist_max 0.141→0.125m, net-laps_min 2.69→2.71 across all shapes. (The earlier original-data DAgger overfit was from piling extra passes onto the SAME few seeds; spreading each pass over a fresh seed range + the clean soft baseline avoids it.) Final soft policy = soft + DAgger×2 (runs_soft_all2), on par with original v2 but from a cleaner base. STAR GENERALIZATION CONFIRMED (2026-07-15): added a `star` shape to shape_dataset (SHAPE_SIDES['star']=10, STAR_INNER_RATIO=0.45 -- a 5-pointed concave star, 10 alternating outer/inner-radius vertices, built with the same angle-increasing arc-length machinery). The soft+DAgger×2 policy, which trained on ONLY triangle/square/pentagon/circle, traverses the never-seen star BOTH directions (net laps 3.36 CCW / 3.39 CW, dist ~0.14m -- slightly higher than trained shapes' ≤0.125m because a star has 2x the corners, but full completion). Visual (star_test.png) shows it tracing all 5 points + 5 concave notches. This is strong evidence that the pos_err+lookahead (path-RELATIVE, no absolute coords / no per-shape features) state design genuinely generalizes to arbitrary paths -- the project's core goal (follow an arbitrary path from state alone, unlike pure-pursuit which needs the whole path in advance).

Training speed (benchmarked 2026-07-15, don't chase GPU for this): the IQL net is a tiny MLP (hidden 256), so **CPU single-thread is the fastest** (~431 it/s) -- CPU all-cores (~199) and Apple MPS (~114) are BOTH slower because parallel/GPU launch overhead dwarfs the small matmuls. So the original `torch.set_num_threads(1)` was correct. main.py now has `--device {auto,cpu,mps,cuda}` (auto = cuda>cpu; MPS excluded from auto since it's slower here, reachable via `--device mps`) and `--threads` (default 1). To honor the override, iql.py was changed to reference `util.DEFAULT_DEVICE` dynamically (it used to `from .util import DEFAULT_DEVICE`, capturing the import-time value, which caused a model-on-mps / data-on-cpu mismatch). On a real CUDA box `--device auto` picks cuda automatically, but GPU only clearly wins if the net/data are scaled up a lot. Everything from "SOLVED" to the end of this paragraph-block is kept only as a record of what was tried, NOT as a working result.

100k-DATA REVIVAL -- DAgger matters more than data volume (2026-07-16). Trained on just 100k steps (33 episodes, head of the soft dataset) with NO DAgger: the policy is STUCK (net laps 0.5-0.6, wanders, doesn't trace any shape -- too few precise-tracking samples to even lock onto the path). Then added a SMALL DAgger pass (n-seeds 20 = 80 episodes, both dirs, corner kicks) merged in (episode_id offset to avoid collision) and retrained: it REVIVES to full traversal (net laps 2.66-3.25 both directions, all 4 shapes traced -- visual confirmed). So completion is decided by DAgger (corner-recovery + covariate-shift fix), NOT by initial-data volume: 100k is plenty as a base, a wandering policy comes back to life with one small DAgger round. Reinforces the whole project's through-line: the look-ahead state feature makes it learnable, and DAgger makes it complete.

STANDALONE INFERENCE + GITHUB PUBLICATION (2026-07-16). The policy is a pure function (path-relative 16-dim state -> 3-dim target_vel), so it runs WITHOUT the simulator or training data -- only the run dir's 3 files (final.pt + config.json + obs_normalization.npz). Added `policy_infer.py`: `PolicyController(run_dir, path)` then `.get_velocity(drone_pos, quat, vel, angvel) -> target_vel` every control step; it computes pos_err/look-ahead from the waypoint path, applies the SAME normalization + slew cap + look-ahead the training used (all three REQUIRED or it diverges), and you can `set_path()` at runtime (unlike pure-pursuit, no full plan needed). Repo github.com/subsubli/drone_simulation now ships: (a) THREE gzipped datasets via LFS -- `data/merged1.5M.csv.gz` (248MB, original 4-shape bidirectional), `data_soft/merged1.5M_soft.csv.gz` (266MB, soft), `data_square_cw/merged_square_cw_1M.csv.gz` (178MB, square CLOCKWISE-only 1M, same soft kicks) -- LFS free tier is 1GB so everything is gzipped (per-episode shape_dataset kept LOCAL-only as *.tar.gz, gitignored); (b) the TWO final policies (final.pt+config+npz) -- `runs/merged/07-15-26_14.49.56_fbpq` (v2) and `runs_soft_all2/merged/07-15-26_18.02.37_uhsf` (soft); (c) README section 7 on running the bundled policies + each dataset's `DATASET_HANDOFF.md` (schema/units/min-max/deviation-distribution/training-benchmark). So a fresh clone can either retrain from scratch OR run the shipped policies OR import PolicyController into its own code.

SQUARE-CW SINGLE-SHAPE INITIAL-ONLY test (2026-07-16). Trained 300k on the square-CW-only 1M dataset (265 eps, soft kicks 0.1/2/0.3, off-path 5.1%), WITHOUT DAgger. Seed 500: square-CW (the trained direction) net laps 0.93 / dist 2.31m = NOT completing (corner-stalled); square-CCW (UNtrained direction) net laps 2.01 -- HIGHER than the trained CW. Conclusion: concentrating 1M steps on ONE shape+direction still does NOT complete without DAgger (off-path 5% = too few corner-recovery samples), and single-shape/single-direction focus does NOT specialize (CCW>CW here; seed-500's CW-square corner just happened to be a harder one). Reconfirms that DAgger -- not data volume, not shape/direction focus -- is the completion-decider (same lesson as the 100k revival). Expect a small DAgger pass would revive it too (not yet run; user said initial-only for now).

SQUARE-CW + 20-EP DAGGER = REVIVED (2026-07-16). First diagnosed the initial-only stall via a closest_idx trace: the policy flew lap 1 at normal speed (idx 0->2779 in 1150 steps) then FROZE at idx 2779 (the last corner) for the remaining 2500 steps (idx range 2779~2779) -- a true STUCK, not slow (more rollout time would not help). Then added a tiny DAgger pass: collect_dagger on the square-CW policy, `--shapes square --direction cw --n-seeds 20` (corner kicks 1.0/8/1.5) = 20 eps; re-merge (265 original + 20 dagger = 285 eps, 1.08M rows) -> retrain 300k (22:12:40, prue). Result seed 500 CW: net laps 0.93 -> **2.73** = REVIVED. closest_idx now crosses the corner and wraps (2963->502 lap-transition, keeps going through ~3 laps) instead of freezing. So a SINGLE shape+direction needs only ~20 DAgger episodes to clear its corner stall (vs 80 eps for the 100k all-4-shapes revival). Final confirmation: corner completion is gated by DAgger (corner-recovery labels), not by data volume or shape/direction concentration.

SINGLE-SHAPE (square-CW) -> ALL-SHAPES GENERALIZATION (2026-07-16) -- the strongest generalization result in the project. Took the square-CW + 20-ep DAgger policy (prue), which saw ONLY ONE shape (square) in ONE direction (CW), and rolled it over 5 shapes x both directions at seed 500. Result: **10/10 traverse** (net laps 2.41-3.59) -- including the never-trained triangle/pentagon/circle, the untrained CCW direction, AND the never-trained star (ccw 3.00 / cw 3.59). Distance is looser than the 4-shape uhsf policy (triangle 0.15m, star 0.21m vs uhsf 0.10/0.14m) since only square was seen, but completion is total. Interpretation: the pos_err+lookahead state is fully shape- AND direction-invariant, so ONE shape+direction is enough to learn the closed-loop "follow the local path signal" behavior, and DAgger supplies the corner-recovery that makes it robust -- together they generalize to arbitrary shapes/directions and an untrained concave star. Visualized: sqcw_dagger_5shapes_ccw.png (2D top-down) + interactive 3D. (Numbers in the results table below.)

TILT-90 (VERTICAL-PLANE) GENERALIZATION (2026-07-16). Added `fixed_tilt_deg` to shape_dataset.run() to force the shape's plane tilt (training used RANDOM ±30° only). Forced tilt=90° (a fully VERTICAL plane -- large z-excursion, completely out of the trained tilt range) and tested BOTH the soft-4shape (uhsf) and square-CW (prue) policies over 4 shapes, seed 500 CCW. Both TRAVERSE: net laps soft-4shape 2.34/2.61/2.95/2.08, square-CW 2.39/2.66/2.83/2.05 (tri/sq/pent/circ), dist 4-9cm. Slightly lower net laps than tilt-30 (circle lowest ~2.05 -- a vertical circle is hardest) but full completion, and the two policies are nearly identical. So the policy is invariant not just to shape and direction but ALSO to PLANE TILT: pos_err+lookahead are 3D world-frame vectors, so a tilted/vertical path presents the same local-signal structure regardless of orientation. Net: the design is shape- AND direction- AND tilt-invariant. (Numbers in results table 10.)

DIFFUSION DATA AUGMENTATION -- pipeline built + key sampling bug fixed (2026-07-16). Goal: augment the offline-RL data with generated transitions. Two approaches, in the new top-level `diffusion/` folder (next to IQL-PyTorch-main/ and gym_pybullet_drones/; generated CSVs under diffusion/gen_*):
- `transition_diffusion.py` (SynthER-style, **REJECTED**): DDPM on individual (s,a,r,s') 36-dim vectors written as 2-row mini-episodes. drone_dataset then reads the 2nd row (s', action=0, done=True) as a FAKE terminal that would teach "action=0 at s'" — contaminates training. Abandoned.
- `trajectory_diffusion.py` (Diffuser-lite, **KEPT**): DDPM with 1D temporal conv over fixed H=64-step trajectory windows [s(16)|a(3)] per step; each generated window = ONE per-episode CSV (H rows), reward recomputed as −|pos_err|, done only on the last step. Consumed by the EXISTING merge_shape_dataset + drone_dataset unchanged (file==episode), NO fake transitions (action |mean| 13.9, not 0).

**Key bug + fix**: generated samples first DIVERGED even though the DDPM loss converged (0.999→0.12). Cause = the reverse sampling had no clamp, so x drifted out of the normalized range and accumulated. Fix = clamp x to [−3,3] (normalized) each reverse step. Remaining quality gap: pos_err is still over-spread (diffusion under-sharpens soft's narrow precise-tracking distribution) — needs more steps / bigger model / full 1.5M (runs so far: H=64, ch=64 temporal conv, T=1000, 8–10k steps on CPU; GPU recommended).

=== DETAILED EXPERIMENT TIMELINE (for report writing; timestamps = run-dir folder names, local time) ===
All training: IQL, hidden 256, beta 3.0, tau 0.85, 300k steps unless noted, `--include-lookahead --reward-clip-min -1.0`, CPU single-thread (~430 it/s, ~12 min/run). All eval on HELD-OUT seeds (not in training). Metric = net laps (target 3; ≥~2.5 = full traversal) via progress_metric.py (coverage% + unwrapped closest-idx laps), plus distance error (mean |pos_err| vs pure-pursuit expert) via evaluate_trained_policy.py. "8/8" = 4 shapes × 2 directions all traverse.

-- 2026-07-15 (original / bidirectional pipeline) --
* Bidirectional data collection: `collect_shape_dataset --direction both --perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5 --att_d_gain_scale 0.3`, 4 shapes, 1.5M steps -> 423 episodes (212 CCW / 211 CW, ~106/shape), off-path(|pos_err|>0.2m) = 78.8%, median |pos_err| 3.33m, max 38m.
* 14:04:05 (akop) INITIAL train on data/merged (1.5M). 
* DAgger-1: collect_dagger over 4 shapes, `--n-seeds 60 --direction both --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 --slew-max-accel 2.0` = 240 eps; re-merge (data + dagger1) = 663 eps / 2.36M rows.
* 14:25:32 (xuyy) RETRAIN v1. Eval seed 500 both dirs: circle & pentagon traverse (2.6-3.35) but SQUARE FAILS both dirs (net laps 1.02 CCW / 0.85 CW) and triangle-CW fails (0.85) -- low-tilt sharp-corner stall.
* DAgger-2: collect_dagger `--shapes square triangle --seed-start 60 --n-seeds 60 --direction both` (kicks 1.0/8/1.5) = 120 eps; re-merge (data+dagger1+dagger2) = 783 eps / 2.78M rows.
* 14:49:56 (fbpq) RETRAIN v2 = FINAL ORIGINAL POLICY. Eval seed 500 both dirs: net laps triangle 2.69/2.67, square 2.90/2.96, pentagon 3.21/3.31, circle 2.70/2.69 = 8/8 traverse; distance error ≤ 12.4 cm (square went 1.8-2.5m -> 0.124m). Multi-seed sweep (500-503, 16 rollouts): dist_mean 0.080m / dist_max 0.124m / net-laps_min 2.67.
* no-lookahead ablation (50k, same data, WITHOUT --include-lookahead): net laps mean −0.14 / min −0.59 (WANDERS/reverses), dist ~0.34m (near path but no progress). Confirms look-ahead is essential (+2.89 -> −0.14 without it).
* Kick-spiral diagnosis: kick=0.3-1.5m but |pos_err| reaches 5-38m; measured cos(raw_target_vel, return-dir)=0.99 (raw already returns at full speed 1.4) while slew crushes commanded |tv| to ~0.44 -> slow return -> spiral at corners. Added `adaptive_lookahead_k` (no effect: raw already correct) and `adaptive_slew_k` (relaxing slew DOES restore |tv|=1.4 and the sim tracks it, but causes OVERSHOOT so net-laps doesn't improve, e.g. triangle-1 0.55->0.19). Both default k=0. Conclusion: keep deviations small via soft data instead.

-- 2026-07-15 (soft dataset) --
* Soft data collection: `--perturb_prob 0.1 --perturb_count 2 --perturb_magnitude 0.3 --direction both` (count is the strongest lever; mag doesn't bound max deviation because it's the recovery trajectory, not the kick), 1.5M -> 423 eps (212/211), off-path 6.8%, median |pos_err| 0.006m (6mm!), max 5.7m. 93% precise-tracking.
* 17:14:13 (rtfx) soft INITIAL-only train: net laps CCW tri 1.87 / sq 2.08 / pent 2.44 / circ 1.75; CW tri 1.66 / sq 0.72 / pent 0.76 / circ 0.69 = NOT completing (too little off-path to learn corner recovery + no DAgger).
* soft DAgger-1 (4 shapes, n-seeds 60, both dirs, kicks 1.0/8/1.5) = 240 eps; re-merge = 663 eps / 2.36M.
* 17:37:53 (vxjv) soft RETRAIN v1: 8/8 traverse (net laps 2.67-3.31) -- clean baseline needed only ONE DAgger pass vs original's TWO.
* soft DAgger-2 (all 4 shapes, seed-start 60, n-seeds 60, both dirs) = 240 eps; re-merge = 903 eps / 3.21M.
* 18:02:37 (uhsf) soft RETRAIN v2 = FINAL SOFT POLICY. Multi-seed sweep (500-503): dist_mean 0.080m / dist_max 0.125m / net-laps_min 2.71 -- IMPROVED over DAgger-1 (0.093->0.080), NO overfit (spread each DAgger pass over a fresh seed range).
* STAR generalization: added `SHAPE_SIDES['star']=10, STAR_INNER_RATIO=0.45` (5-pointed concave star, 10 corners). Soft policy (trained on tri/sq/pent/circ only) traverses star seed 500 both dirs: net laps 3.36 CCW / 3.39 CW, dist 0.142/0.144m. Ratio check: policy/expert dist = star 3.9x vs triangle 3.3x vs circle 1.4x -- star's larger absolute error is because it has 2x the corners (expert itself is 0.036m on star vs 0.005m on circle), NOT a generalization failure.

-- 2026-07-16 (100k revival + square-CW data) --
* 17:41:10 (mdjv) 100k tiny-initial: took the head 100k rows of soft data = 33 episodes, trained 300k WITHOUT DAgger. STUCK: net laps CCW tri 0.64 / sq 0.54 / pent 0.62 / circ 0.55 (wanders, traces no shape -- visualized).
* small DAgger on the 100k policy: `--n-seeds 20` = 80 eps; merged with the 100k head (episode_id offset) = 113 eps / 383k rows.
* 18:08:58 (tchy) 100k+DAgger RETRAIN: REVIVES to net laps CCW tri 2.71 / sq 2.94 / pent 3.23 / circ 2.79; CW 2.66 / 2.91 / 3.25 / 2.80 = 8/8 traverse (visualized: all 4 shapes traced). => completion decided by DAgger, not initial-data volume; 100k is a plenty-large base.
* square-CW dataset: `--shapes square --direction cw --target_steps 1000000` with soft kicks (0.1/2/0.3) = 265 eps / 1.004M steps, 265 CW / 0 CCW. Pushed as data_square_cw/merged_square_cw_1M.csv.gz (178MB LFS).
* Cleanup: kept 2 final policies (fbpq, uhsf) + PNGs + data_square_cw; gzipped all datasets (original 630->248MB, soft, square-CW), per-episode shape_dataset -> local *.tar.gz (gitignored); pushed policies + policy_infer.py + README §7.
* 21:27:46 square-CW single-shape initial-only (no DAgger, 300k on the square-CW 1M data, off-path 5.1%): seed 500 CW (trained dir) net laps **0.93** / dist **2.31m** = NOT complete (corner-stalled); CCW (untrained) net laps 2.01 (> trained CW). Single-shape+direction focus does NOT replace DAgger; off-path 5% too few for corner recovery -- same lesson as 100k revival.
* 22:12:40 square-CW + 20-ep DAgger: closest_idx trace showed the initial-only failure was STUCK (idx frozen at 2779, the last corner, for 2500 steps -- not slow). Added DAgger 20 eps (square, CW) → re-merge 285 eps / 1.08M → retrain. seed 500 CW net laps **0.93 → 2.73** = revived (idx now wraps past the corner through ~3 laps). A single shape+direction needs only ~20 DAgger eps to clear its corner stall.
* 22:44 same square-CW+DAgger policy (prue) rolled over 5 shapes × both directions (seed 500): 10/10 traverse (net laps 2.41-3.59) incl untrained triangle/pentagon/circle, untrained CCW, and never-trained star (ccw 3.00 / cw 3.59). One shape+one direction of training generalizes to all shapes+directions (numbers in Results table 9).
* tilt-90 (vertical-plane) test: added `fixed_tilt_deg` to sd.run; both the soft-4shape and square-CW policies traverse all 4 shapes at tilt=90° (net laps 2.05-2.95, dist 4-9cm) despite training on ±30° tilt only → the policy is TILT-invariant too (3D world-frame state). Table 10.
* diffusion augmentation (diffusion/ folder): built transition-diffusion (SynthER-style, rejected — 2-row episodes give fake action=0 terminals) and trajectory-diffusion (Diffuser-lite, kept — H=64 windows, clean merge/load). Generation first DIVERGED (action max 85) despite converged DDPM loss; fixed by clamping the reverse sampling to [−3,3]. After clamp: action |mean| 1.60 / pos_err 0.90m / vel 2.88 — all in real range (table 11). pos_err still over-spread (under-sharpened narrow distribution) → needs more steps/bigger model/full data. WHY THE KICK-RECOVERY SPIRAL happens (measured, resolves "kick 0.3m but |pos_err| reaches 5-38m"): after a kick the pure-pursuit RAW target_vel already points almost perfectly back to the path (cos(raw,return)=0.99) at full speed 1.4 -- so LOOK-AHEAD is NOT the problem. The SLEW-RATE cap throttles the COMMANDED speed to ~1/3 (|tv|=0.44) while it rotates prev_target_vel from "forward" to "return", so recovery is slow; the drone coasts on inertia meanwhile and at a corner this compounds into an outward spiral. Tried two adaptive fixes in shape_dataset.py (both default k=0 = off, backward compatible): `adaptive_lookahead_k` (shrink look-ahead ∝ deviation) -- NO effect, because raw was already returning; `adaptive_slew_k` (relax slew cap ∝ deviation) -- the command DOES swing to full-speed return instantly (0.44→1.4) and the sim physically tracks it (no blow-up), BUT it causes OVERSHOOT so net-laps doesn't improve (triangle-1 0.55→0.19). Normal cases stay intact for both. Conclusion: recovering a large deviation without overshoot is intrinsically hard for pure-pursuit + inertia (an expert limitation); the practical fix is to keep the deviation SMALL in the first place (soft perturbation), not to retune the tracker. GITHUB: pushed original `data/merged1.5M.csv` (630MB, LFS) + `data_soft/merged1.5M_soft.csv.gz` (266MB gzip -- LFS free tier is 1GB, so soft is gzipped) + a `DATASET_HANDOFF.md` beside each (schema, units, min/max, deviation-intensity distribution, shape-generation method, 5×5×5m workspace). Repo: github.com/subsubli/drone_simulation.

-- 2026-08-06 (diffusion quality push) --
* 12:09–12:49 (gen_traj_quality) trajectory DDPM upgraded and rerun for QUALITY: MPS device, ch 64→128 + depth 3→6 residual/GroupNorm (0.52M params), EMA 0.999, warmup(1k)→cosine LR 2e-4→1e-5 (logged per line), full 1.5M (92,457 H=64 windows), 50k steps, 500 gen episodes. Loss 2.4→~0.015. GEN vs REAL: pos_err |mean| 0.051 (med 0.022, max 2.09) vs real 0.191 (med 0.006, max 5.66); vel 0.94 vs 1.00; action 1.00 (max 2.79) vs 1.10 (max 1.40); quat 1.000. => vel/action now match real; pos_err |mean| 18× closer than §11's 0.90m. Residuals: GEN median still ~4× real's (mild under-sharpen) and GEN truncates the rare corner-overshoot tail (max 2.09 < 5.66) while action sometimes exceeds the 1.40 slew clip. Full numbers in results table 12.

-- 2026-08-06 (roughness investigation: what makes GEN pos_err jittery — 4 hypotheses tested) --
Motivation: a viz (velocity-integrated flown path vs flown+pos_err reference) showed the generated pos_err carries high-frequency STEP-TO-STEP jitter that real precise-tracking windows don't. Roughness metric = mean ‖pos_err[t+1]−2pos_err[t]+pos_err[t−1]‖ (pe_jerk). Real precise-tracking pe_jerk ≈ 0.00135; the DDPM-quality generator ≈ 0.0086–0.0090 (~6–7× rough). Also added an ACTION physical clamp: scale generated |action| down to the real max (1.40 m/s, the slew cap) — removed the max-2.79 overshoot cleanly (dir preserved).
* 13:10 (gen_traj_cons) pos_err↔vel CONSISTENCY loss (λ=0.5): on the predicted x0 (denormalized), v_target = Δpos_err/dt + vel should have low step-to-step change (smooth path); weighted by ᾱ_t (x0 confidence). cons fell 0.97→0.0001, diff unchanged (0.017, same as no-cons) → added "for free", but roughness only 0.0090→0.0084 (−6%). Not the lever.
* 14:10 λ SWEEP {0.5, 5, 50} (12k steps): roughness 0.0255 / 0.0242 / 0.0384 — does NOT decrease with λ; λ=50 WORSE on everything (diff 0.033, roughness 0.038, pos_err med 0.16). Raising λ is not the lever; user's "cons drops too fast" hypothesis rejected. cons hits ~0.0001 (model makes x0_pred smooth easily) yet samples stay rough → the constraint on x0 can't control final-sample roughness.
* 14:49 (gen_traj_ddim) DDIM DETERMINISTIC sampling (η=0, 100 steps) to remove the ancestral sampler's per-step noise injection (my own hypothesis): at full 50k, DDPM 0.00844 vs DDIM 0.01018 — DDIM NOT smoother (slightly worse) and narrows the distribution (vel 0.74/action 0.77 vs real 1.0/1.1, mode-seeking). Sampler noise is not the lever either. (Added a checkpoint save `model.pt` so samplers can be re-run without retraining.)
* 15:44 ABLATION of the two remaining levers — pos_err **asinh normalization** (arcsinh(pos_err/0.05): linear near 0 so the 6mm bulk stays resolvable, log-compresses the 5.7m tail so per-channel std isn't tail-dominated) vs **dilated receptive field** (conv dilation 1,2,4,8,16 lifts RF from ~33 past H=64). 15k steps, DDPM, else identical. Full-set pe_jerk: baseline 0.02036 / **asinh 0.00327 (6.2× ↓, ~2.4× real)** / dilated 0.02189 (no help) / asinh+dilated 0.00453 (dilation adds roughness back). CONCLUSION: **asinh normalization is THE lever** — the roughness was a representation problem (heavy tail crushing the precise-tracking bulk resolution), exactly as diagnosed; receptive field is irrelevant (roughness is high-freq/local, not long-range). Caveat: asinh(0.05) also compresses the generation of large deviations (GEN pos_err max dropped to ~0.10m) — may need a larger scale to preserve the corner-recovery tail. Levers ruled out: consistency-λ, sampler (DDPM/DDIM). Numbers in results table 13.
* 18:07–21:17 DIFFUSION AUGMENTATION EFFECT experiment. Added `--load` resample mode (generate from a saved model.pt with no retraining) + `--gen-batch`; resampled a 24,000-window pool (1.5M rows) from the asinh generator. Built 3 mixes at fixed 1.5M total (build_mix.py, symlink episodes): real1.0M+diff0.5M, real0.5M+diff1.0M, diff1.5M(pure). Ran the full soft recipe on each IN PARALLEL (run_aug_pipeline.sh, run-dir captured from main.py's "Log dir:" print so the shared runs/ dir has no ls-td race), then eval_aug.py (seeds 500-549 × both dirs = 100 rollouts/shape, validated vs the soft baseline). RESULT (50-seed): completion is perfect (400/400) at both EXTREMES incl PURE diffusion, but the 50/50 blend is unstable (377/400 — held-out blow-ups on triangle/square/pentagon); precision degrades ∝ diffusion fraction (circle dist 0.007→0.019→0.036→0.048). => diffusion substitutes for real on completion (even 100%), not on precision, and a half-and-half blend is worse than either pure endpoint. Full table 15 (the generated data's own error distribution is table 14). (Bug caught & fixed mid-run: initial run_aug_pipeline used `ls -td runs/merged` to find each stage's policy, but main.py names the subdir by csv-stem — d1_merged/d2_merged — so it grabbed the init policy for DAgger-2; switched to capturing the exact "Log dir:" path.)
* 16:39–17:26 (gen_traj_asinh) full 50k CONFIRM run, asinh(0.05) alone (DDPM, 500 gen): full-set pe_jerk **0.00339** (vs §12 baseline 0.00900, real 0.00135 → from 6.7× to 2.5× real, 2.7× smoother); pos_err med **0.0183** (baseline 0.022, real 0.006) / mean 0.036; vel 0.92, action 0.93 (max 1.40, clamped), quat 1.000. Viz (gen_asinh_vs_real.png): the flown+pos_err reference now hugs the ∫vel path smoothly, near-indistinguishable from a real window (GEN |pos_err| 0.009/0.014 vs REAL 0.008/0.011). Confirmed residual: asinh compresses the deviation tail — GEN pos_err max 0.96m vs real 5.66m (few large corner-recovery windows). Net: jitter fixed, precise-tracking realism high; heavy-tail recovery under-represented (tune asinh scale up if the augmentation needs it).

Pipeline is documented in ~/drone_simulation/README.md (1.5M-step recipe: collect -> merge -> train -> DAgger(with corner kicks) -> retrain -> progress+distance eval -> viz), and the reusable eval/viz scripts now live in the examples dir: progress_metric.py (coverage/net-laps -- THE metric), evaluate_trained_policy.py (distance error), viz_paths.py (top-down PNG), viz_paths_3d.py (interactive 3D). All take `<RUN_DIR> [seed]`.

Critical state-design reminder: `target_pos_x/y/z` (or a relative form like `target_pos - pos`) must be included in whatever "state" is fed to the offline-RL policy -- it is NOT optional metadata. Without it the policy has no way to know which direction to move (same raw pos/vel/rpy/ang_vel state maps to different correct actions depending on where the shape's path actually is), making the learning problem ill-posed. Pure per-episode bookkeeping that's safe to exclude from state: `episode_id, step, t, shape, center_x/y/z, start_yaw_deg, tilt_deg, tilt_axis_deg`.

How to apply: when resuming this project, read shape_dataset.py directly for the exact current implementation (this memory may drift from the code over time); use this memory mainly for the *why* behind design choices and the deferred TODO list. Before any real-hardware flight test, treat both (a) attitude-gain retuning for this velocity-only mode and (b) matching `max_speed`/`max_accel` to the real drone's actual thrust-to-weight as prerequisites, independent of how good the trained RL policy is.

---

# Measured Results (tables)

All rollouts on held-out seed 500 (not in training) unless noted. Metric = **net laps** (target 3; ≥~2.5 = full traversal) and **distance error** = mean |pos_err| (m) vs the pure-pursuit expert. "8/8" = 4 shapes × 2 directions all traverse. fbpq/uhsf rows are freshly re-measured; deleted intermediate runs use logged values.

## 1. Final policies — completion + distance (seeds 500–509 × both directions = 20 rollouts/shape, normal random tilt)

soft = soft data + DAgger×2 (`uhsf`); orig = original data + DAgger×2 (`fbpq`). Traverse = net laps ≥ 2.0. **net laps** mean±std (min, traverse-count) and **distance error (m)** mean±std. Expert distance ≈ 0.005–0.03m.

| shape | **soft** laps (min, trav) | **orig** laps (min, trav) | **soft** dist | **orig** dist |
|---|---|---|---|---|
| triangle | 2.70±0.02 (2.65, 20/20) | 2.67±0.06 (2.44, 20/20) | 0.100±0.010 | 0.097±0.009 |
| square | 2.98±0.03 (2.91, 20/20) | 2.70±0.76 (0.21, **18/20**) | 0.114±0.008 | **0.366±0.903** |
| pentagon | 3.28±0.04 (3.21, 20/20) | 3.12±0.62 (0.43, **19/20**) | 0.109±0.007 | **0.183±0.344** |
| circle | 2.76±0.02 (2.72, 20/20) | 2.64±0.08 (2.38, 20/20) | 0.007±0.001 | 0.013±0.003 |
| star *(untrained)* | 3.24±0.16 (3.00, 20/20) | 3.21±0.12 (3.00, 20/20) | 0.145±0.009 | 0.138±0.009 |

**Key finding (only visible with the seed sweep): soft = 100/100 traverse, orig = 97/100.** soft completes every shape on every seed with tiny variance; orig has **intermittent held-out blow-ups** on square (18/20) and pentagon (19/20). The clean soft baseline is markedly MORE ROBUST across seeds — single-seed(500) numbers hid this. (star is untrained for both → generalization.)

## 2. Final policies — distance error (m; seeds 500–509 × both dirs, 20 rollouts/shape)

mean±std (max). Expert reference ≈ 0.005–0.03m.

| shape | **soft** mean±std (max) | **orig** mean±std (max) |
|---|---|---|
| triangle | 0.100±0.010 (0.127) | 0.097±0.009 (0.118) |
| square | 0.114±0.008 (0.128) | 0.366±0.903 (**4.17**) |
| pentagon | 0.109±0.007 (0.128) | 0.183±0.344 (1.68) |
| circle | 0.007±0.001 (0.009) | 0.013±0.003 (0.022) |
| star *(untrained)* | 0.145±0.009 (0.173) | 0.138±0.009 (0.149) |

soft's error is tight and low-variance everywhere; orig's square/pentagon variance is huge (max 4.17m) from the same intermittent blow-ups. On seeds where both complete the precision is comparable (~3–4× expert on corners, ~1.5–2× on the circle) — the difference is entirely orig's occasional failures, i.e. robustness, not average precision.

## 3. Multi-seed sweep (seeds 500–503, 16 rollouts each)

| policy | dist_mean | dist_max | net-laps_min |
|---|---|---|---|
| v2 (orig, DAgger×2) | 0.080 | 0.124 | 2.67 |
| soft DAgger×1 | 0.093 | 0.141 | 2.69 |
| soft DAgger×2 (final) | 0.080 | 0.125 | 2.71 |

Soft DAgger-2 improved on DAgger-1 (0.093→0.080) with NO overfit across held-out seeds.

## 4. DAgger progression — completion (net laps)

| stage | data | tri | sq | pent | circ | verdict |
|---|---|---|---|---|---|---|
| **soft** initial-only | 1.5M, no DAgger | 1.87/1.66 | 2.08/**0.72** | 2.44/**0.76** | 1.75/**0.69** | NOT complete |
| soft + DAgger×1 | +240 eps | ~2.7–3.3 both | ✓ | ✓ | ✓ | 8/8 |
| soft + DAgger×2 (final) | +240 eps | 2.71/2.69 | 2.94/2.98 | 3.29/3.31 | 2.78/2.77 | 8/8 |
| **orig** v1 (DAgger×1) | 2.36M | ok | **1.02/0.85** | ok | ok | square fails |
| orig v2 (DAgger×2, final) | 2.78M | 2.71/2.69 | 2.92/2.96 | 3.31/3.35 | 2.70/2.69 | 8/8 |
| **100k** initial-only | 100k, no DAgger | 0.64/– | 0.54/– | 0.62/– | 0.55/– | STUCK |
| 100k + small DAgger | +80 eps | 2.71/2.66 | 2.94/2.91 | 3.23/3.25 | 2.79/2.80 | 8/8 (revived) |

(cells shown as CCW/CW; bold = failing.) Pattern: completion is gated by DAgger, not by initial-data volume.

## 5. Ablation — look-ahead removed (50k, no `--include-lookahead`)

| | net-laps mean | net-laps min | dist |
|---|---|---|---|
| with look-ahead | +2.89 | +2.67 | ~0.08 m |
| **without** look-ahead | **−0.14** | **−0.59** | ~0.34 m (near path, no progress) |

Look-ahead (lx/ly/lz) is essential: without it the policy wanders/reverses.

## 6. Star generalization (untrained shape, soft policy, seed 500)

| dir | net laps | policy dist | expert dist | policy/expert |
|---|---|---|---|---|
| CCW | 3.36 | 0.142 | 0.036 | 3.9× |
| CW | 3.39 | 0.144 | 0.036 | 4.0× |

Ratio comparable to trained shapes (triangle 3.3×, circle 1.4×) → star's larger absolute error is its 2× corner count, not a generalization failure. A never-trained shape is fully traversed.

## 7. Dataset statistics (1.5M each, both directions)

| | original | soft |
|---|---|---|
| perturbation | prob 1.0 / count 6 / mag 1.5 | prob 0.1 / count 2 / mag 0.3 |
| on-path (≤0.2m) | 21.2% | **93.2%** |
| off-path (>0.2m) | 78.8% | 6.8% |
| median \|pos_err\| | 3.33 m | **0.006 m** |
| max \|pos_err\| | 38.1 m | 5.7 m |
| episodes (CCW/CW) | 423 (212/211) | 423 (212/211) |

## 8. Kick-recovery spiral — adaptive-slew sweep (triangle, kicked episode)

| adaptive_slew_k | net laps | pos_err max (m) |
|---|---|---|
| 0 (off) | 0.55 | 2.39 |
| 2 | 0.77 | 2.69 |
| 5 | 0.77 | 4.33 |
| 10 | 0.19 | 2.69 |

Relaxing the slew cap makes the raw return command reach full speed instantly (|tv| 0.44→1.4, physically tracked) but causes OVERSHOOT, so net-laps doesn't improve — corner recovery without overshoot is an intrinsic pure-pursuit+inertia limit. Diagnosis: after a kick, raw target already points back at cos 0.99; the slew cap (not look-ahead) throttled the return.

## 9. Single-shape (square-CW) policy → all-shapes generalization (seed 500)

Compared side-by-side: **(A) square-CW-only** policy (trained on ONE shape + ONE direction: square-CW 1M + 20-ep DAgger) vs **(B) soft 4-shape** policy (uhsf: triangle/square/pentagon/circle both directions + DAgger×2). For (A), every shape except square-CW is generalization (never in training); for (B), all 4 shapes were trained (both directions) and only the star is untrained. net laps (target 3) / distance error (m):

| shape·dir | **(A) square-CW only** laps / dist | **(B) soft 4-shape** laps / dist |
|---|---|---|
| triangle-ccw | 2.50 / 0.152 | 2.71 / 0.099 |
| triangle-cw | 2.63 / 0.089 | 2.69 / 0.114 |
| square-ccw | 2.97 / 0.176 | 2.94 / 0.125 |
| square-cw | 2.73 / 0.156 | 2.98 / 0.128 |
| pentagon-ccw | 3.19 / 0.166 | 3.29 / 0.115 |
| pentagon-cw | 3.19 / 0.093 | 3.31 / 0.111 |
| circle-ccw | 2.75 / 0.016 | 2.78 / 0.007 |
| circle-cw | 2.41 / 0.027 | 2.77 / 0.006 |
| star-ccw (both untrained) | 3.00 / 0.213 | 3.36 / 0.142 |
| star-cw (both untrained) | 3.59 / 0.125 | 3.39 / 0.144 |

**Both traverse 10/10** (net laps 2.4–3.6). The takeaway of the comparison: completion is essentially the same for both — even the single-shape (A) completes every shape/direction — but **(B) is more PRECISE** (distance ~0.10–0.14m vs A's ~0.09–0.21m; e.g. star 0.14 vs 0.21m, circle 0.006 vs 0.016m). So seeing more shapes doesn't change WHETHER it completes (the shape/direction-invariant state + DAgger already give that from one shape), it tightens HOW closely it tracks — more shape variety = better corner precision, not better completion. **Also notable: (A) tracks TIGHTER in its TRAINED direction (CW).** 4 of 5 shapes have CW dist < CCW dist for (A): triangle 0.089<0.152, square 0.156<0.176, pentagon 0.093<0.166, star 0.125<0.213 (circle is the lone exception, 0.027 vs 0.016). So direction-specialization shows up in the ERROR — the policy is more precise in the direction it actually trained on — even though COMPLETION is direction-agnostic (both directions traverse). (B), trained in both directions, is by contrast roughly symmetric between CCW/CW. Visualized in `sqcw_dagger_5shapes_ccw.png`.

## 10. Tilt-90 (vertical-plane) generalization (seed 500, CCW)

Training used random plane tilt **±30°** only; here tilt is forced to **90°** (fully vertical plane — large z-excursion, well outside the trained range). Two policies: soft-4shape (uhsf) and square-CW (prue). net laps (target 3) / distance (m):

| shape | soft-4shape laps / dist | square-CW laps / dist |
|---|---|---|
| triangle | 2.34 / 0.095 | 2.39 / 0.071 |
| square | 2.61 / 0.076 | 2.66 / 0.082 |
| pentagon | 2.95 / 0.075 | 2.83 / 0.087 |
| circle | 2.08 / 0.043 | 2.05 / 0.062 |

**Both traverse** (net laps 2.05–2.95, dist 4–9cm) even though the plane tilt is 3× the training max — the pos_err+lookahead state is 3D world-frame, so a vertical/tilted path presents the same local signal. circle is the hardest (vertical circle, ~2.05). The two policies are near-identical, so completion here is again from the invariant state + DAgger, not from training-shape variety. Conclusion: the policy is **shape-, direction-, AND tilt-invariant**.

## 11. Diffusion augmentation — generation quality (trajectory diffusion on soft data)

DDPM (1D temporal conv, H=64, ch=64, T=1000, ~8–10k steps) over soft-data trajectory windows. Loss converged (0.999→0.12) in both runs; the divergence was purely a missing sampling clamp.

| metric | clamp OFF | **clamp ON** | real (soft) |
|---|---|---|---|
| action \|mean\| (max) | 10.9 (85) | **1.60 (2.56)** | ±1.4 |
| pos_err \|mean\| (max) | 6.5m (67) | **0.90m (2.68)** | median 0.006, max 5.7 |
| vel \|mean\| | 19.8 | **2.88 m/s** | mostly <2 |
| quat_norm | 1.000 | 1.000 | 1.0 |

Clamping the reverse process to [−3,3] (normalized) each step brought every channel into the real physical range. Remaining gap: pos_err is over-spread (0.90m vs real median 0.006) — the diffusion under-sharpens soft's narrow precise-tracking distribution; to be closed with more steps / a bigger model / the full 1.5M. Pipeline is otherwise clean: each generated trajectory is one per-episode CSV → merge_shape_dataset → drone_dataset with no fake transitions.

## 12. Diffusion augmentation — quality push (2026-08-06 12:49) — MPS + bigger model + full 1.5M

Closed most of §11's gap. Same trajectory DDPM, upgraded: **MPS device** (Apple GPU, ~18 steps/s vs CPU single-thread), **bigger/deeper model** (ch 64→128, 3→6 residual conv blocks + GroupNorm, 0.52M params), **EMA weights** (0.999, sampled from the shadow), **warmup→cosine LR** (1k-step warmup 0→2e-4, cosine→1e-5, printed every log line), **full 1.5M** (92,457 windows, no `--limit`), **50k steps**, 500 generated episodes. Loss 2.4→~0.012–0.02 (vs §11's 0.12). `diffusion/gen_traj_quality/`.

| metric | §11 clamp ON | **§12 quality** | real (soft) |
|---|---|---|---|
| pos_err \|mean\| (med, max) | 0.90m (—, 2.68) | **0.051 (0.022, 2.09)** | 0.191 mean (med 0.006, max 5.66) |
| vel \|mean\| | 2.88 | **0.94** | 1.00 |
| action \|mean\| (max) | 1.60 (2.56) | **1.00 (2.79)** | 1.10 (1.40) |
| quat_norm | 1.000 | 1.000 | 1.0 |

vel and action now essentially match real (0.94 vs 1.00, 1.00 vs 1.10); pos_err |mean| dropped 18× (0.90→0.051) and its median (0.022) is the same order as real (0.006). Two honest residuals: (1) GEN's bulk median (0.022) is still ~4× real's (0.006) — slightly under-sharpened; (2) GEN **truncates the corner-overshoot tail** — GEN pos_err max 2.09 < real 5.66, and real's high mean (0.191) comes from that rare heavy tail GEN under-samples; conversely GEN action occasionally exceeds the slew clip (max 2.79 > real 1.40). So the generator reproduces the tight precise-tracking regime well and the rare large-overshoot corners less so. LR schedule is now visible in `diffusion/quality_run.log`.

## 13. Diffusion — pos_err jitter: 4 levers tested, asinh normalization is the fix (2026-08-06)

Residual (1) above (under-sharpened pos_err) is a step-to-step JITTER, measured by **pe_jerk** = mean ‖pos_err[t+1]−2·pos_err[t]+pos_err[t−1]‖ (real precise-tracking ≈ **0.00135**). Four hypotheses tested, each with a controlled run:

| lever | config | pe_jerk (↓) | verdict |
|---|---|---|---|
| — | DDPM quality baseline (§12) | 0.00900 | reference (~6.7× real) |
| consistency loss | λ=0.5 (pos_err↔vel, on x0) | 0.00844 | −6% only |
| consistency loss | λ sweep 0.5/5/50 (12k) | 0.0255/0.0242/0.0384 | **not the lever** (λ=50 worse) |
| sampler | DDIM η=0, 100 steps (50k) | 0.01018 | **not the lever** (slightly worse, narrows dist) |
| receptive field | dilated conv 1,2,4,8,16 (15k) | 0.02189 vs base 0.02036 | **no effect** (roughness is high-freq/local) |
| **normalization** | **asinh(pos_err/0.05) (15k)** | **0.00327 vs base 0.02036** | **THE lever: 6.2× ↓, ~2.4× real** |
| normalization+RF | asinh + dilated (15k) | 0.00453 | dilation adds roughness back — drop it |

**Root cause = representation, not sampler/architecture/loss-weight.** pos_err's heavy tail (max 5.7m) dominates its per-channel std, compressing the 6mm precise-tracking bulk into <1% of the normalized range, so MSE eps-training can't resolve the fine temporal structure. `arcsinh(pos_err/0.05)` is linear near 0 (keeps the bulk resolvable) and log-compresses the tail (std no longer tail-set) → the model resolves the fine structure and jitter drops 6.2×. Applied consistently in preprocessing, the consistency loss, generation, and stats; inverse `0.05·sinh(·)`. Also added: a physical **action clamp** (scale |action| down to the real 1.40 m/s slew cap — kills the max-2.79 overshoot) and a **model.pt checkpoint** (resample without retraining). Open caveat: asinh(0.05) also compresses generation of large deviations — a larger scale may be needed to keep the corner-recovery tail. (15k-step ablation numbers above; **full 50k asinh confirm run: pe_jerk 0.00339 (2.5× real, 2.7× smoother than §12), pos_err med 0.0183, but tail compressed to max 0.96m vs real 5.66m** — see timeline 16:39.)

## 14. Diffusion DATA — the generator's own tracking-error distribution (2026-08-06 23:24)

Before asking what a policy trained on it does (§15), this is what the generated DATA itself looks like: |pos_err| over the full pure-diffusion 1.5M pool vs the real soft 1.5M it was trained on.

| ‖pos_err‖ (m) | median | mean | p90 | p99 | max | off-path (>0.2m) |
|---|---|---|---|---|---|---|
| **diffusion (pure 1.5M)** | **0.018m** | 0.029 | 0.037 | 0.390 | 1.01 | 1.34% |
| real soft 1.5M | **0.006m** | 0.190 | 0.091 | 3.884 | 5.66 | 6.82% |

Two differences, both consequences of the §13 asinh generator:
1. **The precise-tracking bulk is ~3.2× looser** — diffusion median 18mm vs real 6mm. asinh fixed the step-to-step *jitter* (§13) but the absolute tracking level the generator reproduces is still ~3× real's; it can't reproduce the ultra-tight 6mm bulk.
2. **The tail is far shorter** — diffusion p99 0.39m / max 1.01m vs real p99 3.88m / max 5.66m; off-path 1.34% vs 6.82%. asinh(0.05) compresses the heavy tail, so the generator rarely emits large corner-recovery deviations.

This is the mechanism behind §15: a policy inherits the 18mm data precision (and compounds it in closed-loop rollout → ~48mm circle), while the missing recovery tail (1.34% off-path) is what the DAgger passes have to supply.

## 15. Diffusion AUGMENTATION effect — mix real+diffusion, train full soft recipe, eval like Table 1 (2026-08-06 21:17)

Does the asinh generator's data actually help/replace real data downstream? Built 3 mixed datasets at a fixed 1.5M-row total (real soft episodes + diffusion pool episodes from `gen_traj_asinh/model.pt` via `--load`), then ran the **identical full soft recipe on each** (initial 300k IQL train → DAgger-1 4 shapes seeds 0-59 → retrain → DAgger-2 square+triangle seeds 60-119 → retrain), and evaluated exactly like Table 1 (seeds 500-509 × both dirs = 20 rollouts/shape; `eval_aug.py`, validated to reproduce the soft baseline). "diffusion episodes" are 64-step path-relative windows (NOT whole shapes — the state is shape-agnostic), so they add generic local path-tracking, not shape/corner structure (that gap is exactly what the DAgger passes fill).

Numbers below are the **50-seed** sweep (seeds 500-549 × both dirs = **100 rollouts/shape**, 400/policy; expanded from an initial 10-seed pass to tighten the stats and catch rare held-out blow-ups). Held-out is safe: data collection used seeds ≤422, DAgger ≤119.

| shape | **soft** real1.5M | **real1.0M+diff0.5M** | **real0.5M+diff1.0M** | **diff1.5M (pure)** |
|---|---|---|---|---|
| triangle laps (min, trav) | 2.71±0.02 (2.65, 100/100) | 2.67±0.02 (2.63, 100/100) | 2.39±0.66 (0.30, **87/100**) | 2.60±0.05 (2.46, 100/100) |
| square | 2.99±0.03 (2.90, 100/100) | 2.90±0.06 (2.79, 100/100) | 2.74±0.50 (0.29, **95/100**) | 2.80±0.09 (2.58, 100/100) |
| pentagon | 3.30±0.05 (3.20, 100/100) | 3.20±0.05 (3.04, 100/100) | 3.02±0.51 (0.39, **95/100**) | 3.10±0.10 (2.85, 100/100) |
| circle | 2.76±0.03 (2.57, 100/100) | 2.69±0.03 (2.56, 100/100) | 2.63±0.03 (2.55, 100/100) | 2.49±0.02 (2.43, 100/100) |
| **TOTAL traverse (4 trained)** | **400/400** | **400/400** | **377/400** | **400/400** |
| *star (UNTRAINED) laps (min, trav)* | 3.24±0.34 (1.00, 98/100) | 3.27±0.30 (1.56, 99/100) | 3.17±0.61 (0.43, **93/100**) | 3.20±0.36 (0.74, 99/100) |
| circle dist (m) | **0.007** | 0.019 | 0.036 | **0.048** |
| triangle / square / pentagon dist | 0.104 / 0.116 / 0.109 | 0.114 / 0.134 / 0.126 | 0.311 / 0.234 / 0.207 | 0.114 / 0.132 / 0.125 |
| star dist (m) | 0.165 | 0.164 | 0.256 | **0.151** |

**Two findings (the 50-seed sweep sharpened both):**
1. **Completion is preserved at the EXTREMES — including 100% diffusion (400/400)** — pure-diffusion initial data + DAgger traverses every shape, both dirs, all 50 held-out seeds, rock-solid. This re-confirms "DAgger decides completion, not initial-data volume/quality" and shows diffusion data is a viable substitute for real data on the completion axis. BUT the **50/50 blend (real0.5M+diff1.0M) is the unstable one — 377/400**, with clustered held-out blow-ups on triangle (87/100), square/pentagon (95/100), laps dropping to ~0.3 and dist to ~3.3m. Non-monotonic: both pure endpoints are perfect, the middle is not — mixing a real heavy-tail distribution with the diffusion's compressed-tail one seems to give a more conflicting training signal than either alone. (The 10-seed pass only hinted at this — 78/80 — the 50-seed makes it systematic.)
2. **Precision degrades ∝ diffusion fraction** — cleanest on circle (no corners, pure precision test): dist **0.007 → 0.019 → 0.036 → 0.048** (7× at pure diffusion), monotonic and independent of the completion story. The policy inherits the generator's looser precise-tracking (gen median pos_err 0.018m vs real 0.006m). Absolute error stays small (≤5cm circle; corners ~11-13cm except the 50/50 mix's blow-up-inflated means), and circle net laps drift down 2.76→2.49.

3. **Untrained-shape generalization survives augmentation** — the never-trained 5-pointed star still traverses on all policies (soft 98/100, mix1/mix3 99/100), and pure diffusion (mix3) even gives the tightest star distance (0.151±0.011). The 50/50 blend is again the weakest (93/100), consistent with its instability on the trained shapes. So the diffusion-augmented policies keep the shape-invariant generalization.

**Headline:** diffusion augmentation **keeps completion AND untrained-shape generalization (DAgger-carried, even at 100% diffusion) but trades tracking precision ∝ how much real data it replaces; a 50/50 real+diffusion blend is less stable than either pure endpoint**. Diffusion stands in for real data to answer "does it complete the shape," not "how tightly does it track." Tooling: `build_mix.py` (episode-level real+diff mixing via symlinks), multi-folder `merge_shape_dataset.py`, `run_aug_pipeline.sh` (per-mix full recipe, run-dir captured from main.py's print → parallel-safe), `eval_aug.py` (Table-1-style laps+distance over a configurable seed sweep).

## 16. Initial-only (DAgger ABLATION) — same 4 datasets, NO DAgger (2026-08-06 23:40)

The §15 policies all had DAgger×2. This evaluates each mix's INITIAL-only policy (300k train on the mix, no DAgger) to separate "initial-data-mix effect" from "DAgger contribution". 10 seeds (500-509) × both dirs = 20 rollouts/shape (moderate, per the ablation's purpose).

| shape (init-only) | soft real1.5M | real1.0M+diff0.5M | real0.5M+diff1.0M | diff1.5M (pure) |
|---|---|---|---|---|
| triangle laps (min, trav) | 1.73±0.55 (0.01, 8/20) | 0.35±0.26 (0.00, 0/20) | 0.45±0.35 (0.00, 0/20) | 1.04±0.59 (0.02, 0/20) |
| square | 1.67±0.62 (0.29, 7/20) | 0.45±0.39 (0.00, 0/20) | 0.49±0.32 (0.00, 0/20) | 1.18±0.74 (0.14, 2/20) |
| pentagon | 1.69±0.70 (0.49, 8/20) | 0.38±0.26 (0.00, 0/20) | 0.46±0.32 (0.00, 0/20) | 1.14±0.72 (0.00, 5/20) |
| circle | 1.47±0.48 (0.39, 1/20) | 0.45±0.29 (0.02, 0/20) | 0.43±0.27 (0.04, 0/20) | 1.36±0.35 (0.40, 0/20) |
| **TOTAL traverse (init-only)** | **24/80** | **0/80** | **0/80** | **7/80** |
| same policy **+ DAgger×2** (from §15, per-100 scaled) | ~400/400 | 400/400 | 377/400 | 400/400 |

**Two findings:**
1. **Nothing completes without DAgger** — best is pure real soft at only **24/80 (30%)**; the diffusion-containing sets are 0–7/80. DAgger then lifts *every* dataset to ~400/400. This is the cleanest direct proof of the project's core claim: **DAgger, not the initial data (volume, realness, or mix), decides completion.**
2. **Both blends collapse to 0/80 — strictly worse than either pure endpoint** (soft real 24/80, pure diffusion 7/80). The "mixing a real heavy-tail distribution with the diffusion's compressed-tail one is worse than either alone" effect — visible as the 50/50 instability *after* DAgger in §15 — is far starker *before* DAgger: both mixes are the worst, and pure diffusion actually out-tracks the real-containing blends (laps ~1.1–1.4 vs ~0.4). (init-only ⇒ all stuck/off-path, so the distances here just reflect being stranded, not precision.)

**Stuck vs slow vs lost — why the low init-only laps differ by dataset** (closest-idx trace + off-path distance, `eval_stuck.py`, 3 seeds × both dirs, averaged over the 4 shapes). "lastQ advance" = laps the closest-path-index still gains in the final quarter of the episode (≈0 ⇒ frozen); "dist" = mean |pos_err|:

| init-only policy | net laps | coverage | lastQ advance | dist (m) | verdict |
|---|---|---|---|---|---|
| **soft real1.5M** | 1.61 | 0.66 | 0.30 | 0.53 | **SLOW** (tracing, would complete w/ more time) |
| **diff1.5M (pure)** | 1.27 | 0.61 | 0.28 | 0.44 | **SLOW** (tracing, ~as good as pure real) |
| real1.0M+diff0.5M | 0.37 | 0.26 | ~0.00 | 3.4 | **LOST** (diverged off-path, frozen) |
| real0.5M+diff1.0M | 0.40 | 0.30 | ~0.01 | 3.0 | **LOST** (diverged off-path, frozen) |

low net laps does NOT mean the same failure across datasets:
- **soft real1.5M and diff1.5M(pure) init are SLOW, not stuck** — coverage ~0.6, index still advancing at episode end, |pos_err| < 1m. They genuinely trace the path, just ~2× too slowly to close 3 laps in the fixed episode.
- **both real+diffusion BLENDS are LOST (diverged)** — coverage ~0.25, index frozen, |pos_err| **3–4m off the path**. Tracking has collapsed, not merely slowed. So the "mixing is worse than either pure source" effect isn't a slowdown — the blend breaks path-following outright, while pure diffusion init tracks about as well as pure real init.

So the §15/§16 "mixing is worse than either pure source" is sharper than the traverse counts alone: a pure source (real or diffusion) yields a slow-but-valid tracker before DAgger, whereas a 50/50-ish real+diffusion blend yields a policy that diverges off the path entirely. Pure diffusion init tracks about as well as pure real init (both SLOW, dist <1m); it's the *combination* of the real heavy-tail and the diffusion compressed-tail distributions that breaks initial tracking.

## 17. CLASS-CONDITIONAL generation — split on-path vs off-path so the modes stop blending (2026-08-07 00:26)

§13/§14's open caveat: asinh fixed the jitter but the generator still blends two physically-distinct sub-distributions into one — precise-tracking (bulk ~6mm) and corner/kick recovery (metres off-path) — and that blend is what leaks a smeared tail into otherwise-precise windows (§14 diffusion p99 0.39m even though 98.7% of rows are on-path). Structural fix: **condition the diffusion on a per-window on/off-path class** (0=on-path window with max|pos_err|≤0.2m, 1=off-path window that contains a recovery excursion; same 0.2m threshold as tables 7/13/14), so the two modes are learned separately and can be generated on demand. Implemented in `trajectory_diffusion.py` (`--class-cond`): a class embedding added the same broadcast way as the timestep embedding (n_classes=0 ⇒ byte-identical old behavior, pre-class checkpoints load unchanged), classifier-free-guidance training (`--cond-dropout 0.1` relabels to a null class) and sampling (`--cfg-weight`), plus class-balanced batch sampling (`--offpath-batch-frac 0.5`, multinomial reweight — natural 7% starves the off-path branch) and a generation split knob (`--gen-offpath-frac`, default = the data's natural ratio). Training input is the **unchanged** soft 1.5M CSV; the label is derived at load time from the existing pos_err columns (no re-collection). Full run: 46,301 windows (7.0% off-path = the table-6.8% consistency check), ch=128 depth=6 asinh(0.05) λ_cons=0.1 offpath-batch-frac 0.5 cfg 1.5, 50k steps MPS ~40min.

**Mode separation is clean and complete** (`pe_dist.py`, 0.2m threshold, `--gen-offpath-frac` selects the mode):

| ‖pos_err‖ (m) | median | mean | p90 | p99 | max | off-path (>0.2m) |
|---|---|---|---|---|---|---|
| **cc on-path (frac 0)** | 0.0162 | 0.017 | 0.030 | **0.042** | **0.07** | **0.00%** |
| diffusion asinh, unconditioned (§14) | 0.018 | 0.029 | 0.037 | 0.390 | 1.01 | 1.34% |
| real soft 1.5M | 0.006 | 0.190 | 0.091 | 3.884 | 5.66 | 6.82% |
| **cc off-path (frac 1)** | 0.993 | 0.972 | 1.020 | 1.030 | 1.04 | 100% |
| **cc natural-mix (frac 0.07)** | 0.0165 | 0.086 | 0.038 | 1.019 | 1.04 | 7.00% |

1. **The on-path mode now has ZERO tail leakage** — p99 0.042m vs the unconditioned generator's 0.390m (**9× cleaner**), max 0.07m vs 1.01m, off-path 0.00%. This directly confirms the §14 hypothesis: the smeared tail in "precise" windows was recovery samples bleeding across the un-separated conditional, not a fundamental limit. Generate `--gen-offpath-frac 0` and you get pure precise-tracking data with no excursions. on-path **pe_jerk 0.00212** (vs asinh 0.00296) — smoother, and the 50:50 oversampling did **not** regress on-path precision (the feared backbone-marginal shift didn't materialize, because conditioning carries the separation).
2. **The off-path mode is a clean recovery cluster** — 100% off-path, tightly concentrated ~1.0m (p90–max all 1.02–1.04). On demand you get pure recovery segments.
3. **What class-conditioning did NOT fix** (both are residual asinh(0.05) tail-compression, now clearly localized to the off-path branch):
   - **on-path bulk still ~16mm vs real 6mm** — separating the modes did not tighten the precise-tracking level; median is unchanged from §14's 18mm. The ~3× bulk-looseness is a bulk-resolution limit, not a tail-blend artifact.
   - **off-path mode is a narrow ~1m band, not real's 3–5m tail** — it clusters at 1.0m and never reaches real's p99 3.88m / max 5.66m. asinh(0.05) compresses large deviations inside the off-path branch too, so the generator makes *mild* recovery, not the full-range corner recovery DAgger supplies. A larger asinh scale (or a separately-normalized off-path branch) is the next lever if the recovery tail's *range* matters downstream.

Net: class-conditioning delivered exactly the **mode separation** it targeted (on-path p99 9× cleaner, recovery isolatable), confirming the blend was the tail mechanism — but the two *residual* gaps (16mm bulk, ≤1m recovery range) are asinh-scale limits, not blend artifacts. Checkpoint `gen_traj_cc/model.pt` (class-cond, resample-only via `--load`). Downstream policy effect (does a cc-augmented dataset beat the §15 unconditioned mixes?) not yet run.

**asinh-scale sweep (2026-08-07 01:16, 15k-step class-cond, c ∈ {0.05, 0.2, 0.5}) — raising the scale is a losing trade.** §13's caveat was that a larger asinh scale might restore the recovery tail; tested directly. Absolute values are undertrained (15k vs 50k) so read the trend across scales, not the levels. ON-PATH monotonically DEGRADES with c: pe_jerk(DDPM) 0.00995→0.01146→0.01540, median 0.018→0.031→0.034m, and on-path starts leaking (max 1.00→1.54m, off-path 0.06%→0.49%). OFF-PATH range grows only marginally: p99 0.325→0.429→**0.530m**, max 0.97→1.20m — even c=0.5 gets nowhere near real's recovery p99 3.88m / max 5.66m. So a single global asinh scale cannot satisfy both bulk precision and tail range: c=0.5 costs +55% pe_jerk / +88% median for a recovery p99 that's still 7× short of real. Decision: **keep c=0.05** (the precision sweet spot); the recovery tail is DAgger's job (§15 showed augmentation preserves completion because DAgger supplies corner recovery), not the generator's. If recovery *range* is ever genuinely needed, the right lever is structural (separately-normalized off-path branch), not a global scale bump.

## 18. Class-conditional AUGMENTATION downstream — does §17's clean generator beat §15's unconditioned one? (2026-08-07 02:55)

§17 showed class-conditioning cleanly separates the on/off-path modes at the DATA level (on-path p99 0.042m vs the unconditioned generator's 0.390m). Does that help the POLICY? Regenerated a 24,000-window (1.5M-row) cc pool from `gen_traj_cc/model.pt` at the natural 7% off-path ratio (full-pool check: median 0.017m, p99 1.015m, max 1.04m, off-path 6.97% — the clean-separation composition), then built the **same three mix ratios as §15** and ran the **identical §15 recipe** (init 300k → DAgger-1 4-shape seeds 0-59 → retrain → DAgger-2 sq+tri seeds 60-119 → retrain) and the **same 50-seed eval** (500-549 × both dirs = 100 rollouts/shape, star = untrained generalization).

cc results (§15's unconditioned counterpart in parentheses on the completion/precision rows):

| shape (laps: min, trav) / dist | **real1.0M+cc0.5M** | **real0.5M+cc1.0M** | **cc1.5M (pure)** |
|---|---|---|---|
| triangle | 2.67 (2.63, 100/100) | 2.66 (2.47, 100/100) | 2.65 (2.54, 100/100) |
| square | 2.90 (2.76, 100/100) | 2.88 (2.77, 100/100) | 2.87 (2.74, 100/100) |
| pentagon | 3.18 (1.12, **99/100**) | 3.13 (0.37, **99/100**) | 3.14 (2.95, 100/100) |
| circle | 2.69 (2.64, 100/100) | 2.64 (2.50, 100/100) | 2.53 (2.47, 100/100) |
| **TOTAL traverse (4 trained)** | **399/400** (§15 400/400) | **399/400** (§15 **377/400**) | **400/400** (§15 400/400) |
| circle dist (m) | 0.021 (§15 0.019) | **0.032** (§15 0.036) | **0.040** (§15 0.048) |
| tri / sq / pent dist | 0.113 / 0.133 / 0.146 | 0.114 / 0.137 / 0.141 | 0.110 / 0.128 / 0.123 |
| *star (UNTRAINED): trav / dist* | 100/100 / 0.154 (§15 99/100 / 0.164) | **100/100 / 0.155** (§15 **93/100 / 0.256**) | 100/100 / 0.153 (§15 99/100 / 0.151) |

**Three findings — cc's advantage over the unconditioned generator GROWS with diffusion fraction:**
1. **Precision advantage scales with diffusion reliance.** circle dist (pure-precision axis) — cc vs §15 by mix: 1.0:0.5 → 0.021 vs 0.019 (**tied**, real dominates), 0.5:1.0 → **0.032 vs 0.036**, 0:1.5 → **0.040 vs 0.048** (~17%). The more the policy leans on diffusion data, the more cc's cleaner on-path distribution (§17 p99 0.042 vs 0.39) helps; at high real fraction the real heavy-tail dominates and the two generators are indistinguishable. cc does NOT tighten the on-path *bulk* median (still ~16mm, §17), so the gain is from removing the smeared-tail contamination the policy otherwise compounds in closed loop, not from tighter bulk.
2. **cc nearly cures the 50/50-blend instability §15 flagged.** real0.5+cc1.0 = **399/400 vs §15's 377/400**: §15's clustered blow-ups (triangle 87 / square 95 / pentagon 95) collapse to a **single** isolated pentagon failure. Same story on the untrained star: **100/100 / 0.155 vs §15's 93/100 / 0.256**. cc's clean mode separation makes the diffusion distribution internally consistent, removing the "real heavy-tail + diffusion compressed-tail = conflicting signal" §15 blamed — though one residual pent blow-up means not perfectly cured.
3. **At high real fraction, cc ≈ unconditioned (a wash).** real1.0+cc0.5 = 399/400 with one stray pentagon blow-up (min 1.12, dist 2.12m) and circle 0.021 ≈ §15's 400/400 / 0.019 — within seed noise. Expected: diffusion is a minority there, so which generator produced it barely matters. (Both cc blends carry exactly one isolated pentagon blow-up; pentagon is the sharpest-corner trained shape and the residual weak spot.)

**Headline:** the §17 class-conditional generator is a **better augmentation source than the §15 unconditioned one, and the margin scales with how much diffusion data the mix uses** — tied when real dominates, ~17% tighter precision + fixed instability (377→399/400, star 93→100/100) when diffusion dominates. This is the first result where a **generator-side change (not DAgger)** measurably improved downstream policies. Final policies kept: `mix_cc1.5`, `mix_r0.5_cc1.0`, `mix_r1.0_cc0.5` d2 run dirs; init policy `runs/merged/…_ehsq` kept for the pending initial-only (§16-style) cc ablation.

## 19. Class-conditional INITIAL-ONLY (no DAgger) — the §18 win INVERTS before DAgger (2026-08-07 05:49)

§18 showed cc is a better augmentation source *with* the full DAgger×2 recipe. §16 showed the initial-only (no-DAgger) story for the unconditioned mixes. This runs the same §16 ablation on the three cc mixes: init-only 300k IQL train (cc1.5 & real0.5+cc1.0 inits retrained from their kept `merged.csv`; real1.0+cc0.5 reused the pipeline's init `…_ehsq`), evaluated like §16 (`eval_aug.py` 10 seeds 500-509 × both dirs = 20/shape; `eval_stuck.py` 3 seeds for the stuck/slow/lost verdict).

| init-only (no DAgger) | traverse | mean dist (m) | verdict | §16 unconditioned counterpart |
|---|---|---|---|---|
| **cc1.5 (pure)** | **0/80** | **~21 (16–25)** | **LOST (catastrophic)** | diff1.5-pure: **7/80, SLOW, <1m** |
| real0.5M+cc1.0M | 0/80 | ~2.1 | LOST | real0.5+diff1.0: 0/80, LOST, ~2m |
| real1.0M+cc0.5M | 0/80 | ~2.6 | LOST | real1.0+diff0.5: 0/80, LOST |
| *(ref) soft real1.5M* | *24/80* | *<1m* | *SLOW-but-valid* | *(same policy)* |

**The §18 result inverts: cc's clean separation is a LIABILITY without DAgger, worst at pure.**
1. **Every cc mix is 0/80 LOST** — none traces before DAgger, same as §16's blends. But **pure cc is CATASTROPHICALLY lost (dist ~21m, max 35m)**, far worse than §16's unconditioned pure-diffusion, which was a *slow-but-valid* tracker (7/80, dist <1m). So on the pure-diffusion axis the class-conditional generator is *dramatically worse* init-only than the unconditioned one.
2. **Mechanism — clean separation creates a recovery COVERAGE HOLE.** §17's win was that cc removes the 0.2–1m smeared tail from on-path windows and clusters off-path tightly at ~1m. But that same cleanliness means the training data has almost nothing in the 0.2–1m mid-range and **nothing beyond ~1m** (§17: off-path max 1.04m vs real 5.66m). A DAgger-less init that drifts into that hole has never seen how to return and diverges without bound. The real heavy-tail (3–5m recovery) is exactly what caps the divergence — which is why the real-containing blends stay LOST-at-~2m while **pure cc, with no real recovery data at all, blows out to ~21m**.
3. **This proves the §18 gain is entirely DAgger-enabled.** DAgger drives the policy to its own off-path states and labels them with pure-pursuit's raw recovery answer — supplying exactly the mid/long-range recovery coverage cc's data lacks. Fill the hole (DAgger) and cc's cleaner precise-tracking bulk wins (§18); leave it unfilled (init-only) and cc's hole makes it the worst source.

**Headline:** class-conditioning is a **DAgger-conditional** improvement — it makes the generator a better augmentation source *only because* DAgger backfills the recovery coverage the clean separation removes. Init-only it's strictly worse (pure cc catastrophically so), a sharp new instance of the project's core lesson (DAgger decides completion): here data *cleanliness* and pre-DAgger *robustness* are directly opposed. Kept init policies: `runs/merged/…_tvpq` (cc1.5), `…_vhxz` (real0.5+cc1.0), `…_ehsq` (real1.0+cc0.5).

## 20. GAN as an alternative generator — R3GAN matches/beats diffusion at the data level (2026-08-07)

Built `gan/trajectory_gan.py` as a drop-in alternative to the diffusion generator: SAME data pipeline (`load_windows`), asinh(0.05) normalization, per-window on/off-path class + `--offpath-batch-frac` balancing, pos_err↔vel consistency loss, action clamp, checkpoint/`--load` resample, and identical per-episode CSV output, so `build_mix.py` / `merge_shape_dataset.py` / `eval_aug.py` / `pe_dist.py` consume its pool unchanged. Only the generator swaps: a conditional GAN maps latent z (+class) to an H-step window via the same 1D temporal-conv backbone; the discriminator is a conv+global-pool critic with a Miyato projection term for conditioning.

**Stabilization was the whole battle** (the generator collapsed in every naive setup — one side always dominated):
1. **SN + hinge, symmetric lr** → D over-powers G (lossD→0.06, lossG rising 3→∞), generator diverges (pos_err inf, vel 116).
2. **TTUR (lr_d = lr_g/4)** → over-corrected, D too weak → **mode collapse** (all windows ≈0.6m; a 30k run even collapsed a 10k-quality 15mm bulk to 264m).
3. **Dynamic D-skip** (skip D when its win-rate>0.9) → held the loss balance (d_win pinned 0.9) but a weak D still let G collapse.
4. **R3GAN recipe = RpGAN + R1 + R2 + LR-cosine-decay + best-checkpoint** → STABLE. Relativistic pairing (loss depends only on D(real)−D(fake)) removes the runaway; R1+R2 gradient penalties smooth D; LR decay settles; best-checkpoint (lowest on-path pos_err median, `--eval-every`) captures the peak because GANs are non-monotonic. **on-path median fell monotonically to 8.1mm at step 6500** (past where every earlier setup plateaued/collapsed), then the run still degrades late (median→1m by 9.5k) — but best-checkpoint keeps step 6500, so the saved generator is the peak.

**Data-level comparison — GAN (best ckpt) vs diffusion cc (§17), same asinh/class-cond/offpath-balance:**

| ‖pos_err‖ (m) | median | p99 | max | off-path (>0.2m) | pe_jerk |
|---|---|---|---|---|---|
| **GAN on-path** | **0.0083** | 0.136 | 0.29 | 0.56% | 0.0074 |
| diffusion on-path (§17) | 0.0162 | 0.042 | 0.07 | 0.00% | 0.0021 |
| GAN off-path | 1.017 | 1.029 | 1.03 | 100% | — |
| diffusion off-path | 0.993 | 1.030 | 1.04 | 100% | — |
| GAN natural-mix (7%) | 0.0096 | 1.020 | 1.03 | 7.59% | 0.0075 |
| diffusion natural-mix | 0.0165 | 1.019 | 1.04 | 7.00% | — |
| real soft | 0.006 | 3.88 | 5.66 | 6.82% | 0.00135 |

**Two generators, complementary strengths:**
1. **GAN's precise-tracking bulk is TIGHTER** — on-path median 8.3mm vs diffusion's 16.2mm, approaching real's 6mm. The adversarial objective captures the ultra-tight precise mode that §14 found diffusion's MSE-eps training *couldn't* reproduce (it was stuck ~3× real). This is the GAN's clear win.
2. **Diffusion's separation is CLEANER and smoother** — on-path leak 0.00% vs GAN's 0.56% (a few GAN windows reach 0.29m), p99 0.042 vs 0.136, and pe_jerk 0.0021 vs GAN's 0.0074 (diffusion trajectories are ~3.5× smoother). Iterative denoising produces cleaner tails and less jitter than the GAN's single-shot generation.
3. **Both achieve the class-conditional mode separation** — off-path modes are essentially identical (~1m cluster, 100% off-path), and both hold the 7% natural composition.

**Roughness diagnostic — structural, not undertraining.** Tracked on-path pos_err median and pe_jerk separately over a 10k R3GAN run (both computed on the same on-path sample each eval). The two optimize at essentially the SAME step — **median-best 6500 (0.0078m), pe_jerk-best 7000 (0.00683)** — and pe_jerk falls in lockstep with the median rather than continuing to improve after the bulk is captured. Critically, pe_jerk **plateaus at ~0.0068** across steps 6000-8000 (0.00743→0.00699→0.00683→0.00689→0.00708) before the run collapses, i.e. it converges to a floor ~3× diffusion's 0.0021 and ~5× real's 0.00135 rather than trending toward them. So the roughness is a **structural limit of single-shot GAN generation, not undertraining**: the GAN reaches its roughness floor concurrently with its bulk optimum, and that floor sits well above diffusion's — consistent with §14's point that iterative denoising is what produces smooth trajectories. (Weak confound: the late collapse prevents observing a longer stable region past 7000, but the pre-collapse plateau already indicates floor convergence.) GAN generator kept at `gan/gen_gan_cc/model.pt` (best step 6500). Downstream (pool→mix→policy, §18-style) not yet run.

### 20b. Roughness FIXED — direct smoothness penalty (`--lambda-smooth`) overcomes the GAN floor (2026-08-08)

§20 called the GAN roughness a structural floor. It is a floor for the *plain* objective, but a **direct pos_err 2nd-difference penalty on the generator** (`--lambda-smooth`, targeting pe_jerk itself) pushes straight through it. λ sweep (5k steps each, on-path best): λ=0 → pe_jerk 0.0068; λ=1 → 0.0059 (median 13mm); λ=10 → **0.0025** (median 19mm) ≈ diffusion — a clear smoothness↔bulk frontier the penalty walks. Then, crucially, **step budget matters**: the earlier collapse comes at ~80% of the LR-decay schedule, so a short run peaks early and shallow. Run λ=10 at the **full diffusion budget (50k)** and the peak arrives at step 13000, far better on BOTH axes (best-checkpoint now scores `median+pe_jerk`):

| on-path ‖pos_err‖ (m) | median | p90 | p99 | max | off-path (>0.2m) | pe_jerk |
|---|---|---|---|---|---|---|
| **GAN λ=10, 50k (best step 13000)** | **0.0031** | **0.005** | **0.008** | **0.01** | **0.00%** | **0.00055** |
| GAN plain, §20 (10k) | 0.0083 | 0.030 | 0.136 | 0.29 | 0.56% | 0.0074 |
| diffusion cc, §17 | 0.0162 | 0.030 | 0.042 | 0.07 | 0.00% | 0.0021 |
| real soft | 0.006 | — | 3.88 | 5.66 | 6.82% | 0.00135 |
| GAN λ=10 off-path | 1.036 | 1.040 | 1.040 | 1.04 | 100% | — |
| GAN λ=10 natural-mix (7%) | 0.0035 | 0.007 | 1.040 | 1.04 | 7.00% | — |

**The smoothness-penalized GAN dominates every on-path metric** — median 3.1mm (5× tighter than diffusion, 2× tighter than real), p99 0.008m (5× cleaner tail than diffusion), pe_jerk 0.00055 (4× smoother than diffusion, smoother than real itself), 0.00% leak, perfect mode separation. Penalizing jitter also tightened the bulk (smooth trajectories have less step-to-step noise → lower median). So §20's roughness "structural limit" was really a *missing-objective* limit: once pe_jerk is in the loss, the GAN beats diffusion outright at the data level.

**Caveat — possibly over-idealized.** Being tighter AND smoother than *real* data (3mm/0.00055 vs real 6mm/0.00135) means λ=10 may push past realism into over-smoothed, over-precise trajectories. The pos_err↔vel consistency loss keeps them physically self-consistent, but whether "cleaner-than-real" augmentation data actually helps a downstream policy (vs. just matching real) is an open question for the §18-style eval. A lower λ (e.g. 1: pe_jerk 0.0059, median 13mm — still beats diffusion on bulk, matches on smoothness) is the more conservative choice if realism matters. GAN generator kept at `gan/gen_gan_cc/model.pt` (λ=10, best step 13000).

## 21. GAN AUGMENTATION downstream — cleaner-than-real data HELPS precision (2026-08-08)

The §20b caveat asked whether the λ=10 GAN's over-idealized data (3mm, tighter+smoother than real) helps or hurts a downstream policy. Generated a 24k-window pool from `gan/gen_gan_cc/model.pt` (median 3.3mm, off-path 6.97% — natural composition), built the same three §18 mix ratios, ran the identical soft recipe (init 300k → DAgger×2) and 50-seed eval (500-549 × both dirs, + untrained star).

| shape (laps min, trav) / dist | soft real1.5M (all-real ref) | real1.0M+GAN0.5M | real0.5M+GAN1.0M | GAN1.5M (pure) |
|---|---|---|---|---|
| triangle | 2.71 (2.65, 100/100) | 2.54 (0.16, **92/100**) | 2.70 (2.65, 100/100) | 2.63 (0.35, **97/100**) |
| square | 2.99 (2.90, 100/100) | 2.76 (0.24, **92/100**) | 2.96 (2.86, 100/100) | 2.95 (2.02, 100/100) |
| pentagon | 3.30 (3.20, 100/100) | 3.15 (0.22, **95/100**) | 3.26 (3.18, 100/100) | 3.27 (3.19, 100/100) |
| circle | 2.76 (2.57, 100/100) | 2.64 (0.53, **95/100**) | 2.74 (2.56, 100/100) | 2.77 (2.74, 100/100) |
| **TOTAL traverse (4 trained)** | **400/400** | **374/400** | **400/400** | **397/400** |
| circle dist (m) | **0.007** | 0.098* | **0.009** | **0.008** |
| tri / sq / pent dist | 0.104 / 0.116 / 0.109 | 0.245* / 0.270* / 0.207* | 0.108 / 0.123 / 0.116 | 0.162* / 0.141 / 0.125 |
| *star (untrained): trav / dist* | 98/100 / 0.165 | 97/100 / 0.188 | 99/100 / 0.154 | 100/100 / 0.154 |

(*blow-up-inflated means — a few held-out seeds diverged. soft real1.5M ref = §15/Table 15, the all-real baseline.)

**vs §18 diffusion cc (same ratios): circle dist — real1.0+X0.5: GAN 0.098 vs diff 0.021 · real0.5+X1.0: GAN 0.009 vs diff 0.032 · pure: GAN 0.008 vs diff 0.040.**

**Two findings:**
1. **Cleaner-than-real data HELPS precision — dramatically, the over-idealization worry was wrong.** On the stable mixes the GAN's 3mm data drives circle tracking to **8-9mm — 4-5× tighter than diffusion (21-40mm) and matching the pure-real-soft baseline (7mm)**. The policy inherits the generator's precision, and the GAN's is far higher, so its data makes a *more precise* policy than diffusion's, contradicting the §15/§18 "precision degrades ∝ diffusion fraction" trend (that held for diffusion because its bulk was 3× looser than real; the GAN's bulk is *tighter* than real). Corners on real0.5+GAN1.0 (0.108/0.123/0.116) also beat diffusion cc's (0.114/0.137/0.141).
2. **Stability pattern INVERTS — the high-real blend is the unstable one.** §18's unstable point was the 50/50 blend; here it's **real1.0+GAN0.5 = 374/400** (scattered blow-ups across all four shapes), while real0.5+GAN1.0 is perfect (400/400) and pure GAN near-perfect (397/400, only triangle 97). The GAN distribution is *extremely* far from real (3mm vs 6mm bulk, max 1.04m vs 5.7m tail), so the more real heavy-tail data is mixed in, the sharper the distribution conflict — the opposite balance point from diffusion's milder mismatch.

**Best GAN policy = real0.5M+GAN1.0M**: 400/400, circle **9mm**, corners tighter than diffusion cc, star 99/100 — precision AND stability. Headline: the smoothness-penalized GAN is not just a data-level curiosity — its ultra-precise data yields the **most precise augmented policy in the project** (matching pure-real precision from mostly-synthetic data), at some cost to blend stability when real dominates. Final policies: `mix_gan1.5` / `mix_r0.5_gan1.0` / `mix_r1.0_gan0.5` d2 runs. (Initial-only §19-analog eval still pending — init policies preserved.)

### 21b. GAN INITIAL-ONLY (no DAgger) — the coverage hole is WORST for the cleanest data (2026-08-08)

§19 found cc's clean separation inverts pre-DAgger (all 0/80 LOST, pure cc catastrophic ~21m). The λ=10 GAN data is even cleaner (3mm bulk, off-path a tight 1.04m cluster with NOTHING between 0.01m and 1m and nothing beyond), so this is the sharp test. Init-only eval (§19 protocol, 10 seeds × both dirs; `eval_stuck.py` 3-seed verdict):

| init-only (no DAgger) | traverse | mean dist (m) | max dist | cov | verdict |
|---|---|---|---|---|---|
| **GAN1.5 (pure)** | **0/80** | **8.5–10.6** | **31–47** | 0.07 | **LOST (barely moves, flies off)** |
| real0.5M+GAN1.0M | 0/80 | 2.9–4.3 | 7.8–17 | 0.25 | LOST |
| real1.0M+GAN0.5M | 0/80 | 3.7–7.9 | 15–22 | 0.28 | LOST |
| *(ref) §19 pure cc* | *0/80* | *~21* | — | — | *LOST* |
| *(ref) §16 pure diffusion* | *7/80* | *<1* | — | — | *SLOW-but-valid* |
| *(ref) §16 soft real1.5M* | *24/80* | *<1* | — | — | *SLOW-but-valid* |

**The cleaner the data, the harder the pre-DAgger divergence — monotonic across generators.** Init-only |pos_err|: real/unconditioned-diffusion stay <1m (SLOW-but-valid, some traverse), cc blows to ~21m, and the ultra-clean GAN is the worst yet — **pure GAN reaches 8–10m mean / 47m max with coverage 0.07** (it barely advances one path index before flying off). Mechanism (same as §19, sharper): the GAN's near-perfect separation means the training data has essentially nothing in the 0.01–1m mid-range and nothing past 1.04m, so a DAgger-less init that drifts into that void has zero learned recovery and diverges without bound; real's heavy tail (to 5.7m) is what caps divergence, so the pure (no-real) GAN blows out hardest.

**The DAgger-conditional inversion is now complete and monotonic:** the SAME data ranking flips between the two regimes. WITH DAgger (§21) the cleanest data is BEST (pure/GAN-heavy → circle 8–9mm, most precise policy in the project); WITHOUT DAgger the cleanest data is WORST (pure GAN → 47m divergence). DAgger backfills exactly the mid/long-range recovery coverage that data cleanliness removes — so generator-side precision and pre-DAgger robustness are strictly opposed, and the opposition grows with how clean the generator is (real < diffusion < cc < GAN). Init policies kept (`runs/merged/…_csnw` gan1.5, `…_qlop` r0.5gan1.0, `…_crwe` r1.0gan0.5).

### 21c. 100%-on-path GAN (ZERO recovery data) — the 7% off-path is DAgger's minimum foothold (2026-08-08)

§21b showed cleaner data → harder pre-DAgger divergence. This pushes it to the limit: a 1.5M-row pool of **pure on-path GAN windows (0.00% off-path**, median 3.1mm, max 0.02m — no recovery data at all), then the usual init + DAgger×2. Both stats (p99 = pooled per-step |pos_err| tail, now reported by `eval_aug`):

| shape (laps min, trav) / dist | INIT-only (no DAgger) | FINAL (DAgger×2) |
|---|---|---|
| triangle | 0.22 (0.00, **0/20**), dist 5.0 / p99 **52.6** | 2.33 (0.00, **82/100**), 0.42 / p99 3.10 |
| square | 0.19 (0.04, 0/20), 2.9 / p99 6.9 | 2.60 (0.16, **84/100**), 0.45 / p99 3.73 |
| pentagon | 0.17 (0.00, 0/20), 2.7 / p99 6.5 | 2.97 (0.17, **89/100**), 0.36 / p99 3.55 |
| circle | 0.13 (0.01, 0/20), 3.0 / p99 9.4 | 2.74 (0.73, **98/100**), 0.043 / p90 0.013 / p99 2.36 |
| star (untrained) | — | 3.31 (2.42, 100/100), 0.159 / p99 0.43 |
| **TOTAL traverse (4 trained)** | **0/80** (LOST, cov 0.07, max 67m) | **353/400** |

**Removing the last 7% recovery hurts BOTH regimes:**
1. **init-only is the worst divergence in the whole project** — 0/80, coverage 0.07 (barely advances one index before flying off), triangle p99 **52.6m** / max 67m. With literally zero off-path data the on-path attractor has *nothing* around it, so any drift is unrecoverable — even worse than §21b's 7%-off GAN pure (max 47m). Monotone endpoint of real<diffusion<cc<GAN(7%)<GAN(0%).
2. **DAgger×2 CANNOT fully rescue it — 353/400 vs the 7%-off GAN pure's 397/400.** The trained shapes keep held-out blow-ups (triangle 82/100, square 84/100) that DAgger cleared on every recovery-containing dataset. DAgger injects its own corner-kick recovery, but from a base that ignores the entire off-path space so sharply, two passes can't cover enough of it. **So the 7% off-path recovery in the normal pool was not optional — it is the minimum foothold DAgger builds on.** (Circle survives best at 98/100 because it is corner-free; the polygons' sharp corners are where the missing recovery data bites.)

Contrast with §21: the 7%-recovery GAN pure reached 397/400 and circle 8mm. Strip the 7% and completion falls to 353/400 with corner blow-ups — confirming recovery data and DAgger are complementary, not redundant: DAgger amplifies whatever recovery foundation the data provides, but cannot manufacture it from nothing. Policies: init `runs/merged/…_lryr`, final `runs/d2_merged/…_cusj`; dataset `mix_gan1.5_onpath`.

## 22. CONSOLIDATED downstream reference — completion + precision with p99 tails (2026-08-08)

All downstream policies re-evaluated with pooled per-step |pos_err| percentiles (`eval_aug` now reports p90/p99/pooled-max, not just the per-rollout mean that hid corner-overshoot and blow-up tails). 50 seeds × both dirs = 100 rollouts/shape. "corner" = tri/sq/pent averaged; "circle" = the corner-free pure-precision probe. Key reference table:

| policy (init data) | trav (4-shape) | circle mean | circle **p99** | corner mean | corner **p99** | star trav |
|---|---|---|---|---|---|---|
| **soft real1.5M (all real)** | 400/400 | 0.007 | **0.021** | 0.110 | **0.428** | 98/100 |
| real1.0M+cc0.5M | 399/400 | 0.021 | 0.092 | 0.131 | 0.478 | 100/100 |
| real0.5M+cc1.0M | 399/400 | 0.032 | 0.103 | 0.131 | 0.482 | 99/100 |
| cc1.5M (pure diffusion) | 400/400 | 0.040 | 0.127 | 0.120 | 0.453 | 100/100 |
| real1.0M+GAN0.5M | **374/400** | 0.098 | **3.06** | 0.241 | **3.22** | 97/100 |
| **real0.5M+GAN1.0M** | **400/400** | **0.009** | **0.027** | 0.116 | 0.449 | 99/100 |
| GAN1.5M (pure) | 397/400 | **0.008** | **0.025** | 0.143 | 1.085* | 100/100 |
| GAN 100%-on-path (§21c) | 353/400 | 0.043 | 2.36* | 0.41* | 3.46* | 100/100 |

(*blow-up-inflated — held-out divergences live entirely in the p99/max tail; the p90 beneath is tight, e.g. r1.0GAN0.5 circle p90 0.027, gan1.5 triangle p90 0.389.)

**What p99 reveals that the mean hid:**
1. **Circle (pure precision) is where generators actually differ, and the smoothed GAN wins outright.** circle p99: soft **0.021** ≈ GAN pure/half **0.025/0.027** ≪ diffusion cc **0.092–0.127**. The GAN's circle tail essentially *matches all-real*; diffusion cc's tail is 4–6× worse. So the §20b data-level tightness carries through to the policy: GAN-augmented circle tracking is real-quality, diffusion-augmented is visibly looser even at p99.
2. **Corner shapes are a SHARED structural tail, generator-independent (~0.43–0.48 p99).** Every stable policy — all-real included — overshoots corners to ~0.45m at p99 while its mean sits ~0.11m. This is the pure-pursuit corner-overshoot limit (the expert's own behavior, per §the DAgger notes), not an augmentation artifact: soft real corner p99 0.428 ≈ cc 0.45–0.48 ≈ stable-GAN 0.449. Corner precision is bounded by the expert, so no generator can beat it there — the augmentation choice only moves the circle.
3. **p99 separates instability from precision cleanly.** The unstable mixes (r1.0GAN0.5, gan1.5-triangle, GAN-onpath, even soft's rare star) have tight p90 but p99/max of 1–4m — pure held-out blow-ups, not everyday tracking. Reading means alone conflated these; p90+p99 show r1.0GAN0.5 is actually precise-when-it-works (circle p90 0.027) but occasionally diverges.

**Headline:** on the corner-free precision probe, the smoothness-penalized GAN augmentation delivers **real-quality tracking (circle p99 ~0.025 vs all-real 0.021)** from mostly-synthetic data — 4–6× tighter tails than diffusion — while corner precision is a shared expert-bounded floor no generator moves. The best all-round policy remains **real0.5M+GAN1.0M** (400/400, circle p99 0.027, corners at the structural floor, star 99/100).

### 22 detail — per-shape p99 (pooled per-step |pos_err|, m), all downstream policies

| policy | triangle | square | pentagon | circle | star |
|---|---|---|---|---|---|
| soft real1.5M (all real) | 0.427 | 0.436 | 0.421 | **0.021** | 1.329* |
| real1.0M+cc0.5M | 0.473 | 0.474 | 0.487 | 0.092 | 0.442 |
| real0.5M+cc1.0M | 0.466 | 0.475 | 0.504 | 0.103 | 0.441 |
| cc1.5M (pure diffusion) | 0.451 | 0.466 | 0.443 | 0.127 | 0.440 |
| real1.0M+GAN0.5M | 3.616* | 2.749* | 3.285* | 3.061* | 2.036* |
| real0.5M+GAN1.0M | 0.447 | 0.459 | 0.440 | **0.027** | 0.448 |
| GAN1.5M (pure) | 2.360* | 0.458 | 0.437 | **0.025** | 0.424 |
| GAN 100%-on-path | 3.098* | 3.730* | 3.553* | 2.356* | 0.429 |

(*blow-up-inflated: a few held-out seeds diverged, living in the p99 tail; the p90 beneath is tight. Corner shapes for stable policies cluster at 0.42–0.50 = the shared pure-pursuit corner-overshoot floor; circle is the only column that tracks the generator — soft 0.021 ≈ GAN 0.025–0.027 ≪ diffusion 0.092–0.127. soft's star p99 1.329 is one rare held-out star divergence, not a systematic gap.)

### 22 detail — per-shape p90 (pooled per-step |pos_err|, m), all downstream policies

| policy | triangle | square | pentagon | circle | star |
|---|---|---|---|---|---|
| soft real1.5M (all real) | 0.361 | 0.359 | 0.321 | **0.012** | 0.345 |
| real1.0M+cc0.5M | 0.370 | 0.382 | 0.351 | 0.041 | 0.353 |
| real0.5M+cc1.0M | 0.362 | 0.379 | 0.353 | 0.060 | 0.351 |
| cc1.5M (pure diffusion) | 0.337 | 0.358 | 0.324 | 0.076 | 0.344 |
| real1.0M+GAN0.5M | 0.416 | 0.435 | 0.375 | **0.027** | 0.360 |
| real0.5M+GAN1.0M | 0.370 | 0.374 | 0.333 | **0.016** | 0.346 |
| GAN1.5M (pure) | 0.389 | 0.385 | 0.343 | **0.014** | 0.351 |
| GAN 100%-on-path | 1.762 | 2.296 | 0.460 | 0.013 | 0.350 |

**p90 vs p99 is the instability discriminator.** For the p99-blow-up policies, p90 shows whether the failure is *typical-case* or *rare-tail*: real1.0M+GAN0.5M has tight p90 everywhere (circle 0.027, triangle 0.416 = normal) — it tracks precisely and only *occasionally* diverges (blow-ups confined to p99). GAN 100%-on-path is the opposite — triangle/square p90 are already **1.76 / 2.30m**, so those shapes are *broadly* degraded (the typical rollout is bad, not just the tail), confirming §21c's read that zero-recovery data leaves corner tracking fundamentally unstable rather than merely blow-up-prone. Circle p90 confirms the precision ranking cleanly (no corner tail to muddy it): soft 0.012 ≈ GAN 0.013–0.027 ≪ diffusion cc 0.041–0.076.

### 22 detail — per-shape pooled max (worst single-step |pos_err|, m), all downstream policies

| policy | triangle | square | pentagon | circle | star |
|---|---|---|---|---|---|
| soft real1.5M (all real) | 0.483 | 0.500 | 0.495 | 0.057 | 2.705* |
| real1.0M+cc0.5M | 0.563 | 0.580 | 3.355* | 0.151 | 0.552 |
| real0.5M+cc1.0M | 0.534 | 0.554 | 1.100* | 0.148 | 0.532 |
| cc1.5M (pure diffusion) | 0.527 | 0.563 | 0.552 | 0.194 | 0.534 |
| real1.0M+GAN0.5M | 4.254* | 3.110* | 3.548* | 3.203* | 2.774* |
| real0.5M+GAN1.0M | 0.508 | 0.539 | 0.526 | 0.062 | 1.492* |
| GAN1.5M (pure) | 2.628* | 2.973* | 0.482 | 0.043 | 0.497 |
| GAN 100%-on-path | 4.521* | 4.371* | 4.101* | 2.728* | 2.495* |

(*single held-out seed's worst moment — max is the noisiest statistic, one bad step in 100 rollouts; use it only for absolute worst-case bounds. Stable policies cap corners at ~0.5m / circle ≤0.2m; every value >1m is a lone divergence. Even all-real has one — soft star max 2.705 = the single rare star blow-up also seen in its star p99 1.329, so a >1m worst-case is not unique to synthetic augmentation.)

### 20c. off-path diversity (minibatch-std) — negative: the tail is a DATA limit, not mode collapse (2026-08-09)

The off-path branch collapses to a tight ~1m cluster (§17/§22). Tested whether a StyleGAN2 **minibatch-std** in D (`--mbstd`, punishes batch non-diversity) can widen it toward real's 0.2–5.7m recovery range. Result — it CANNOT reach the tail:

| off-path ‖pos_err‖ | median | p90 | max |
|---|---|---|---|
| baseline (no mbstd) | 1.017 | 1.022 | **1.03** |
| + minibatch-std | 0.847 | 1.003 | **1.03** |
| real soft | — | — | 5.66 |

mbstd spread the cluster a little *within* 0.2–1m (median 1.02→0.85, partially filling the mid-range hole) but **max stayed 1.03m — the 3–5m recovery tail is still absent**, and it *destabilized* on-path (best collapsed at step 2000, combined score 0.021 vs the clean λ=10 run's 0.0035 at step 13000). Net negative.

**Conclusion (confirms the scarcity hypothesis):** the soft dataset has only ~3,228 off-path windows (6.8%) and the 3–5m recovery tail is extremely sparse within them, so single-shot diversity regularization can't *manufacture* a range that's barely in the data — it only redistributes density where data already exists. The off-path range limit is a **data** constraint, not an optimization/mode-collapse one. Widening it would need more off-path data (e.g. the original dirty perturbation dataset, off-path 78.8%), not diversity tricks. GAN generator stays the clean λ=10 config; the `--mbstd` code was removed after this negative result (documented here only).

## 23. Tightening the diffusion bulk — direct x0 pos_err penalty (not min-SNR) (2026-08-09)

§14/§22 left the diffusion generator's one weakness open: a loose precise-tracking bulk (on-path median 16mm vs real 6mm, GAN 3mm), attributed to eps-MSE. Tested two levers at fixed budget. min-SNR-γ weighting (`--min-snr-gamma`, the standard "sample-quality" fix) and a **direct x0 pos_err reconstruction penalty** (`--lambda-x0pe`: `abar · ‖x0_pred_pe − x0_pe‖²`, confidence-weighted to bite at low noise where fine detail is resolved — the diffusion analog of the GAN smoothness term).

30k diagnostic (on-path median / pe_jerk): uniform 0.050 / 0.0027 · min-snr5 **0.022** / **0.0069** · x0pe1 **0.019** / 0.0022. Both tighten the bulk ~2.5×, but **min-SNR roughens** (pe_jerk 0.0027→0.0069 — it down-weights the low-noise timesteps that carry fine detail, exactly the wrong direction for eps-prediction), while **x0pe tightens without the roughness cost**. Full 50k x0pe-1.0:

| on-path ‖pos_err‖ (m) | median | p90 | p99 | max | off-path | pe_jerk |
|---|---|---|---|---|---|---|
| diffusion uniform, §17 (50k) | 0.0162 | 0.030 | 0.042 | 0.07 | 0.00% | 0.0021 |
| **diffusion + x0pe-1.0 (50k)** | **0.0120** | 0.026 | 0.059 | 0.11 | 0.00% | 0.0021 |
| GAN λ=10 | 0.0031 | 0.005 | 0.008 | 0.01 | 0.00% | 0.00055 |
| real soft | 0.006 | — | 3.88 | 5.66 | 6.82% | 0.00135 |

**x0pe tightens the diffusion bulk 16mm → 12mm (~26%) for free — smoothness (pe_jerk 0.0021) and the perfect 0% class separation preserved — but does NOT reach the GAN's 3mm.** The residual ~12mm floor is the eps-MSE mode-averaging limit: the direct penalty softens it but can't eliminate it, whereas the GAN's adversarial objective has no such averaging and hits 3mm. So the honest picture: (1) **my initial min-SNR suggestion was wrong** for this goal — it trades bulk for roughness; (2) the direct x0-pe penalty is the right lever and a genuine modest win, so the *stable* diffusion generator can be made ~25% more precise at zero cost; (3) but ultra-tight bulk (3mm, sub-real) stays a GAN-only capability — adversarial beats denoising-MSE on the finest precise-tracking detail, matching §20b's mechanism. Improved generator kept at `diffusion/gen_traj_cc_x0pe/model.pt`.

### 23b. x0pe diffusion DOWNSTREAM — the data-level gain does NOT reach the policy (2026-08-09)

Ran the §18/§21 downstream protocol on the x0pe-improved diffusion generator (12mm data vs baseline cc 16mm): 24k pool → 3 mixes → init+DAgger×2 → 50-seed+star eval (+ init-only §19-style). Question: does the §23 data tightening carry to the policy?

**FINAL (DAgger×2), circle = pure-precision probe, vs §18/§22 baseline diffusion cc:**

| mix | x0pe trav | x0pe circle mean / p99 | cc trav | cc circle mean / p99 |
|---|---|---|---|---|
| real1.0M+X0.5M | 400/400 | 0.019 / 0.077 | 399/400 | 0.021 / 0.092 |
| real0.5M+X1.0M | 399/400 | 0.030 / 0.087 | 399/400 | 0.032 / 0.103 |
| X1.5M (pure) | 400/400 | **0.049 / 0.153** | 400/400 | **0.040 / 0.127** |

**The 16mm→12mm data gain does NOT produce a meaningful policy gain.** Circle precision is within noise of baseline cc (±0.01) for the real-containing mixes, and **pure x0pe is actually worse (0.049 vs 0.040)**. Completion is comparable (marginally better: 400/400 & 400/400 vs cc's 399/400, within noise). So a modest generator-data improvement washes out in the DAgger + closed-loop pipeline — a 4mm data difference doesn't survive covariate-shift correction and closed-loop compounding. Contrast the GAN (§21): only its *large* data gap (16mm→3mm) moved the policy (circle 0.008); this ~25% gain didn't. **Lesson: generator-data precision only helps downstream past a large threshold; incremental data tightening is not worth the downstream cost.**

**INITIAL-only (no DAgger):** all three 0/80 LOST, divergence mean ~4–10m, max 40–58m — the same coverage-hole behavior as §19 cc (~21m) and §21b GAN. x0pe's clean data (0% on-path leak, off-path a tight ~1m cluster) leaves the same recovery void, so a DAgger-less init diverges hard. Nothing new mechanistically — confirms the clean-data → pre-DAgger-divergence rule holds for x0pe too (real<diffusion<cc≈x0pe<GAN on divergence severity).

**Net verdict on x0pe:** a genuine *data-level* win (§23, stable diffusion 16→12mm) that is **not a policy-level win** (§23b). Keep x0pe as a documented option; the baseline diffusion cc and the GAN λ=10 remain the reference generators. Init policies `runs/merged/…_gomu/_pryh/_yuee`, finals `…_zlzb/_kwrf/_vbpe`.

## 24. Attitude sanity-check — soft (all-real) policy roll/pitch is healthy, no sustained ringing (2026-08-10)

Checked whether the reference soft (all-real) policy `runs_soft_all2/merged/…_uhsf` emits sane attitude under rollout (att_d_gain_scale=0.3, 100Hz control, both dirs). Measured **tilt** = angle between body-z and world-z (yaw-invariant, avoids euler gimbal artifacts), and FFT'd roll/pitch for ringing.

**Tilt (per rollout, one sd.run per fresh process):**

| shape | tilt max (deg) | tilt RMS (deg) | flips (>90°) | pos_err max (m) |
|---|---|---|---|---|
| circle | 11.6 | 5.4 | 0 | 0.02 |
| triangle | 12.6 | 7.9 | 0 | 0.42 |
| pentagon | 12.5 | 8.9 | 0 | 0.43 |
| square | 12.9 | 8.2 | 0 | 0.42 |

Peak tilt ~13°, RMS 6–9°, **zero flips** — matches the known ±~11° velocity-only oscillation (shape_dataset.py:553), already damped by att_d_gain_scale=0.3. Healthy for a real ~2kg drone.

**FFT (roll/pitch power spectrum, Hanning, DC removed):**

| shape | dominant peak | peak Q | 0.8–1.2Hz band power |
|---|---|---|---|
| circle | 0.10Hz | 1.5 | 0.4–0.8% |
| triangle | 0.15Hz | 1.7 | 11–16% |
| square | 0.24Hz | 4.5 | 13–14% |
| pentagon | 0.32Hz | 4.3 | 10–11% |

**No sustained ringing / limit cycle.** The dominant spectral peak is the *maneuvering* frequency (0.1–0.32Hz, scales with corner-passing rate), low-Q (1.5–4.5 = broadband, not a sharp resonance). The ~1Hz attitude mode the comments warn about (the ±11° limit cycle at DEFAULT gains) is **suppressed** by att_d_gain_scale=0.3 — it survives only as a **corner-excited transient**: ~10–16% of (mean-removed) roll/pitch power on cornered shapes, but **<1% on the smooth circle**. So it is not self-sustaining; corners kick a residual 1Hz mode that decays, amplitude bounded (roll RMS 5–6°). 

**Gotcha logged:** running many `sd.run` in ONE process and picking the latest CSV by mtime can read a stale/mismatched file → phantom "180° flip" readings. Diagnose attitude with **one rollout per process**. eval_aug is unaffected (fresh `DSLPIDControl` per rollout so D_COEFF_TOR never compounds; aggregate laps/dist are sane), i.e. no existing eval numbers are corrupted.

**FFT experimental setup & caveats (matters for reading the ringing verdict):**

- **Record length = one full default rollout (3 laps), NOT a controlled steady-state segment.** All at fs=100Hz, one FFT per rollout, Hanning window, DC bin zeroed. Per-shape record length and resulting frequency resolution (Δf = fs/N):

| shape | samples N | duration (s) | Δf = fs/N (Hz) |
|---|---|---|---|
| circle | 2962 | 29.6 | 0.034 |
| triangle | 3306 | 33.1 | 0.030 |
| square | 3727 | 37.3 | 0.027 |
| pentagon | 4053 | 40.5 | 0.025 |

- **Frequency resolution Δf ≈ 0.025–0.034 Hz.** So the low-freq dominant peaks (0.1–0.32Hz) sit on only ~3–13 bins — the exact peak frequency is **coarsely resolved and should not be read as precise**. The 1Hz band of interest, by contrast, is well resolved (~30th bin, the 0.8–1.2Hz window spans ~12 bins), so the **"is there a sustained 1Hz ring?" verdict is valid** at this record length; only the sub-Hz peak *location* is fuzzy.
- **Spectrum is non-stationary.** The whole trajectory (straights + corner accel/decel + laps) is FFT'd together, so corner kicks and maneuvering all mix in — this is *why* the dominant peak reads as a "maneuvering frequency" rather than a clean attitude mode. It bounds ringing (no sustained high-Q peak appears) but is not a pure attitude-loop spectrum.
- **To sharpen if ever needed:** run a longer rollout (e.g. 10 laps ~100s) to ~3× the resolution, or window only straight-segment steady flight to isolate the pure attitude spectrum. Not done here — the current length already answers the sustained-ring question; deferred unless a finer attitude characterization is required.

## 25. Ringing measured on the SOFT SOURCE DATA (not policy) — same signature, plus a 9.2% crash discovery (2026-08-10)

§24 measured *policy rollouts*. Here we FFT the **raw collected soft episodes** directly (`data_soft/shape_dataset/*seed*.csv`, 423 episodes, each a continuous 3-lap expert pure-pursuit rollout at att_d_gain_scale=0.3, 100Hz — same collection settings as eval). Same tilt (body-z vs world-z) + roll/pitch FFT as §24. Read straight from CSV (no sim), so no stale-CSV risk.

**Ringing on healthy episodes (crashed ones excluded, see below), per-shape medians:**

| shape | n (healthy) | tilt max (deg) | tilt RMS (deg) | dominant peak (Hz) | 0.8-1.2Hz band | 1Hz band p90 |
|---|---|---|---|---|---|---|
| circle | 94 | 11.0 | 5.4 | 0.10 | 0.0% | 0.0% |
| triangle | 95 | 11.1 | 6.7 | 0.17 | 12.0% | 15.1% |
| square | 101 | 11.1 | 6.7 | 0.24 | 12.4% | 17.9% |
| pentagon | 94 | 11.1 | 6.8 | 0.29 | 13.8% | 18.4% |

**Verdict — the source data has the same signature as the policy (§24): no sustained ring.** Dominant peak = maneuvering frequency (0.10–0.29Hz, scales with corner rate); the ~1Hz attitude mode is only a **corner-excited transient** (12–14% of mean-removed roll/pitch power on cornered shapes, **0% on the smooth circle**). So the policy did not *introduce* ringing — it faithfully inherited the expert's corner-kicked-but-decaying attitude. att_d_gain_scale=0.3 already suppressed the sustained ±11° 1Hz limit cycle that exists at default gains, in the DATA itself.

**Discovery — 9.2% of soft episodes are real crashes, not ringing.** Scanning all 423 for tilt>90° (yaw-invariant, so a true inversion, not a euler artifact):

| metric | value |
|---|---|
| episodes inverted >1% of steps | 39 / 423 (9.2%) |
| per shape (circle/triangle/square/pentagon) | 11 / 11 / 5 / 12 |
| crash onset (fraction into episode), median | 0.24 (p10–p90 = 0.19–0.29) |
| onset in first 10% of episode | 0 / 39 |
| frozen-inverted tail (vel≈0, upside-down, last 20%) | 25 / 39 |

Raw check of a crashed episode (circle-seed7): normal until ~step 500 (tilt <10°, |pos_err| <0.02m, |vel|~1.3), then by step 1500 it is **tilt 180°, |pos_err| 3.99m, |vel| 0.00** and stays frozen inverted ~4m off-path to the end (reward ≈ −4). A genuine collection crash, not a logging artifact.

**Cause = the perturbation kicks.** Collection used `--perturb_prob 0.1` (10% of episodes get 2 kicks). The kick schedule (shape_dataset.py:595) places the first kick at 0.2 into the episode — and crash onsets cluster exactly at 0.24. So a ~0.3m position kick occasionally destabilizes the velocity-only controller enough to tumble the drone, which then can't recover and freezes inverted. The 9.2% flip rate ≈ the 10% perturb rate.

**Why it matters:** (1) part of the soft set's "off-path" data is **not usable recovery** — it's frozen-crashed tails (drone stuck at 4m, velocity 0), which connects to the off-path scarcity/quality story in §20c and the coverage-hole in §16/§19 (some off-path windows teach "sit still, inverted, far away," the opposite of recovery). (2) The reference soft policy trains *through* this 9% contamination and still works — the path-relative state + windowing + the 90.8% clean majority dominate. (3) Cleaning these 39 crashed tails (or gating kicks to not exceed the controller's recovery envelope) is a concrete, untested data-quality lever if recovery coverage is ever revisited.
