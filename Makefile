NVCC      := nvcc
# -arch=native compiles for the GPU in this machine.
# Swap to e.g. -arch=sm_86 to target a specific architecture (Ampere = sm_80/86).
NVCCFLAGS := -O2 -std=c++17 -arch=sm_89

# -G enables device-side debug info (disables GPU optimisations — debug builds only)
# -g enables host-side debug info
DEBUGFLAGS := -G -g -O0 -std=c++17 -arch=sm_89

# ── MPI build (M4) ─────────────────────────────────────────────────────────
# Use the MPI compiler wrapper as nvcc's host compiler so it picks up the
# MPI include + lib paths automatically. -DUSE_MPI toggles kernels/mpi_comm.cuh
# from no-op stubs to real MPI calls. Open MPI's CUDA-aware path is used by
# default; pass --mpi-host-stage at runtime to force a host-staged fallback.
MPICXX     ?= mpicxx
MPI_NVCCFLAGS := $(NVCCFLAGS) -ccbin $(MPICXX) -DUSE_MPI
MPIRUN     ?= mpirun
MPIRUN_OVERSUB := --oversubscribe

BIN_DIR := build
BINS    := $(BIN_DIR)/camera_projection $(BIN_DIR)/quat_to_rotmat $(BIN_DIR)/colmap_reader_test $(BIN_DIR)/splat_data_test $(BIN_DIR)/simple_knn_test $(BIN_DIR)/projection_2dgs $(BIN_DIR)/projection_2dgs_bwd $(BIN_DIR)/gradient_checks $(BIN_DIR)/train $(BIN_DIR)/loss_test $(BIN_DIR)/intersect_tile $(BIN_DIR)/rasterize_fwd $(BIN_DIR)/adam_test $(BIN_DIR)/hash_grid_test $(BIN_DIR)/mlp_test $(BIN_DIR)/nexel_color_test

.PHONY: all run-camera run-quat run-knn run-adam run-projection-bwd run-gradcheck run-train-live run-viewer run-viewer-points run-viewer-gsplat debug debug-quat debug-train clean run-bench-quick run-bench-nsys run-bench-ncu-fwd run-bench-ncu-bwd run-bench-resolution run-train-mpi-1 run-train-mpi-2 run-train-mpi-4 run-mpi-parity run-bench-mpi-nsys run-bench-strong run-bench-weak

all: $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/camera_projection: kernels/camera_projection.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/camera_projection_debug: kernels/camera_projection.cu | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

$(BIN_DIR)/quat_to_rotmat: kernels/quat_to_rotmat.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/quat_to_rotmat_debug: kernels/quat_to_rotmat.cu | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

run-camera: $(BIN_DIR)/camera_projection
	./$(BIN_DIR)/camera_projection

run-quat: $(BIN_DIR)/quat_to_rotmat
	./$(BIN_DIR)/quat_to_rotmat

$(BIN_DIR)/colmap_reader_test: kernels/colmap_reader_test.cu kernels/colmap_reader.hpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-colmap-test: $(BIN_DIR)/colmap_reader_test
	./$(BIN_DIR)/colmap_reader_test

$(BIN_DIR)/splat_data_test: kernels/splat_data_test.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-splat: $(BIN_DIR)/splat_data_test
	./$(BIN_DIR)/splat_data_test

$(BIN_DIR)/simple_knn_test: kernels/simple_knn_test.cu kernels/simple_knn.cuh kernels/splat_data.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-knn: $(BIN_DIR)/simple_knn_test
	./$(BIN_DIR)/simple_knn_test

$(BIN_DIR)/projection_2dgs: kernels/projection_2dgs.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/projection_2dgs_bwd: kernels/projection_2dgs_bwd.cu kernels/projection_2dgs.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/gradient_checks: kernels/gradient_checks.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu kernels/hash_grid.cu kernels/mlp.cu kernels/nexel_color.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/projection_2dgs_debug: kernels/projection_2dgs.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

run-projection: $(BIN_DIR)/projection_2dgs
	./$(BIN_DIR)/projection_2dgs

run-projection-bwd: $(BIN_DIR)/projection_2dgs_bwd
	./$(BIN_DIR)/projection_2dgs_bwd

run-gradcheck: $(BIN_DIR)/gradient_checks
	./$(BIN_DIR)/gradient_checks

$(BIN_DIR)/train: train.cu kernels/colmap_reader.hpp kernels/splat_data.cuh kernels/simple_knn.cuh kernels/adam.cu kernels/loss.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu kernels/mpi_comm.cuh external/stb_image.h external/stb_image_write.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/train_debug: train.cu kernels/colmap_reader.hpp kernels/splat_data.cuh kernels/simple_knn.cuh kernels/adam.cu kernels/loss.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu kernels/mpi_comm.cuh external/stb_image.h external/stb_image_write.h | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

# Same translation unit, compiled with the MPI wrapper as the host compiler
# and -DUSE_MPI so kernels/mpi_comm.cuh dispatches to real collectives.
$(BIN_DIR)/train_mpi: train.cu kernels/colmap_reader.hpp kernels/splat_data.cuh kernels/simple_knn.cuh kernels/adam.cu kernels/loss.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu kernels/mpi_comm.cuh external/stb_image.h external/stb_image_write.h | $(BIN_DIR)
	$(NVCC) $(MPI_NVCCFLAGS) -o $@ $<

run-train: $(BIN_DIR)/train
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4

run-train-live: $(BIN_DIR)/train
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 --save-ply checkpoints/garden_latest.ply --save-ply-every 50 --densify-start 250 --densify-every 50 --densify-stop 15000

# ── M4 MPI training helpers ──────────────────────────────────────────────
# np=1 is bit-identical to ./build/train (with the default --seed 42).
# np=2 and np=4 oversubscribe the single L4 — useful for correctness
# validation; wall-clock numbers from oversubscribed runs are not meaningful.
#
# MPI_RUNTIME_FLAGS: empty by default. On the AWS-bundled OpenMPI in this
# environment, `opal_built_with_cuda_support` is `false`, so passing a
# device pointer to MPI_Allreduce segfaults inside the vader BTL. The
# helper targets below pass `--mpi-host-stage` to force the D2H→Allreduce
# →H2D fallback. On a node with a CUDA-aware MPI build (mvapich2-gdr,
# OpenMPI with `--with-cuda`, etc.), drop the flag for the faster path.
MPI_RUNTIME_FLAGS ?= --mpi-host-stage

run-train-mpi-1: $(BIN_DIR)/train_mpi
	$(MPIRUN) -np 1 $(MPIRUN_OVERSUB) ./$(BIN_DIR)/train_mpi --data data/360_v2/garden --images images_4 --iters 50 --log-every 10 $(MPI_RUNTIME_FLAGS)

run-train-mpi-2: $(BIN_DIR)/train_mpi
	$(MPIRUN) -np 2 $(MPIRUN_OVERSUB) ./$(BIN_DIR)/train_mpi --data data/360_v2/garden --images images_4 --iters 50 --log-every 10 $(MPI_RUNTIME_FLAGS)

run-train-mpi-4: $(BIN_DIR)/train_mpi
	$(MPIRUN) -np 4 $(MPIRUN_OVERSUB) ./$(BIN_DIR)/train_mpi --data data/360_v2/garden --images images_4 --iters 50 --log-every 10 $(MPI_RUNTIME_FLAGS)

# 50-iter np=1 vs np=2 parity check, asserts losses agree within tolerance.
run-mpi-parity: $(BIN_DIR)/train_mpi
	@bash scripts/test_mpi_parity.sh

# Strong / weak scaling sweeps — wrap the shell scripts so they pick up
# the built train_mpi via the dep. Plots: scripts/bench_comm_breakdown.py
# is invoked manually after the sweep prints its summary.
run-bench-strong: $(BIN_DIR)/train_mpi
	bash scripts/bench_strong_scaling.sh

run-bench-weak: $(BIN_DIR)/train_mpi
	bash scripts/bench_weak_scaling.sh

# Multi-rank nsys timeline. Use after run-mpi-parity passes.
run-bench-mpi-nsys: $(BIN_DIR)/train_mpi
	mkdir -p logs/perf
	$(MPIRUN) -np 2 $(MPIRUN_OVERSUB) \
	    nsys profile -t cuda,nvtx,osrt,mpi --mpi-impl openmpi \
	        --capture-range=nvtx --nvtx-capture=profile_window \
	        -o logs/perf/nsys_mpi_np2_rank%q{OMPI_COMM_WORLD_RANK} --force-overwrite=true \
	    ./$(BIN_DIR)/train_mpi --data data/360_v2/garden --images images_4 \
	        --iters 220 --profile-start-iter 200 --profile-iters 20 \
	        --profile-csv logs/perf/kernel_breakdown_mpi.csv --profile-tag mpi_np2 \
	        --profile-exit --log-every 50

# ── M3 perf benchmarks (see notes/perf_plan.md) ────────────────────────────
# Layer-A breakdown at fresh-init splat count (~140k on garden, no densification).
# Cheap: ~30 sec total. Use for quick sanity checks while iterating on perf tweaks.
run-bench-quick: $(BIN_DIR)/train
	mkdir -p logs/perf
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 \
	    --iters 220 --profile-start-iter 200 --profile-iters 20 \
	    --profile-csv logs/perf/kernel_breakdown.csv --profile-tag fresh_init \
	    --profile-exit --log-every 50

# Layer-A breakdown at steady-state splat count (post densify_stop). Long run
# (~30–60 min on L4) — saves checkpoint at exit so ncu deep-dives can reuse it.
run-bench-steady: $(BIN_DIR)/train
	mkdir -p logs/perf checkpoints
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 \
	    --iters 16100 --profile-start-iter 16000 --profile-iters 100 \
	    --profile-csv logs/perf/kernel_breakdown.csv --profile-tag steady_state \
	    --profile-exit --log-every 500 \
	    --save-ply checkpoints/garden_bench.ply

# Layer-B nsys timeline capture, scoped to the profile_window NVTX range.
# Outputs logs/perf/nsys_<tag>.nsys-rep (open in nsys-ui).
run-bench-nsys: $(BIN_DIR)/train
	mkdir -p logs/perf
	nsys profile -t cuda,nvtx,osrt \
	    --capture-range=nvtx --nvtx-capture=profile_window \
	    -o logs/perf/nsys_fresh_init --force-overwrite=true \
	    ./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 \
	        --iters 220 --profile-start-iter 200 --profile-iters 20 \
	        --profile-csv logs/perf/kernel_breakdown_nsys.csv --profile-tag fresh_init_nsys \
	        --profile-exit --log-every 50

# Layer-C ncu deep-dive on the top kernels. Use --launch-skip to land inside
# the profile window so ncu replays steady-state kernels, not warm-up ones.
run-bench-ncu-fwd: $(BIN_DIR)/train
	mkdir -p logs/perf
	ncu --set full --kernel-name regex:"rasterize_fwd_kernel" \
	    --launch-skip 200 --launch-count 3 \
	    -o logs/perf/ncu_rasterize_fwd --force-overwrite \
	    ./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 \
	        --iters 210 --profile-start-iter 200 --profile-iters 10 \
	        --profile-exit --log-every 50

run-bench-ncu-bwd: $(BIN_DIR)/train
	mkdir -p logs/perf
	ncu --set full --kernel-name regex:"rasterize_bwd_kernel" \
	    --launch-skip 200 --launch-count 3 \
	    -o logs/perf/ncu_rasterize_bwd --force-overwrite \
	    ./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 \
	        --iters 210 --profile-start-iter 200 --profile-iters 10 \
	        --profile-exit --log-every 50

# Layer-D2: resolution sweep — re-run with images_2 / images_4 / images_8.
run-bench-resolution: $(BIN_DIR)/train
	mkdir -p logs/perf
	@for tier in images_2 images_4 images_8; do \
	    echo "▶ resolution sweep: $$tier" ; \
	    ./$(BIN_DIR)/train --data ../data/360_v2/garden --images $$tier \
	        --iters 220 --profile-start-iter 200 --profile-iters 20 \
	        --profile-csv logs/perf/resolution_sweep.csv --profile-tag $$tier \
	        --profile-exit --log-every 50 ; \
	done

run-viewer:
	python3 python/viser_splat_viewer.py --ply checkpoints/garden_latest.ply --port 8080 --resolution 1024

run-viewer-points:
	python3 python/viser_point_cloud_viewer.py --ply checkpoints/garden_latest.ply --port 8080 --poll-seconds 2.0

run-viewer-gsplat:
	python3 python/gsplat_viewer_from_ply.py --ply checkpoints/garden_latest.ply --gsplat-root ../gsplat --port 8081

$(BIN_DIR)/loss_test: kernels/loss.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-loss: $(BIN_DIR)/loss_test
	./$(BIN_DIR)/loss_test

$(BIN_DIR)/adam_test: kernels/adam.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-adam: $(BIN_DIR)/adam_test
	./$(BIN_DIR)/adam_test

$(BIN_DIR)/intersect_tile: kernels/intersect_tile.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ── M5 (Nexels appearance) — hash grid + MLP smoke binaries ───────────────
$(BIN_DIR)/hash_grid_test: kernels/hash_grid.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-hash-grid: $(BIN_DIR)/hash_grid_test
	./$(BIN_DIR)/hash_grid_test

$(BIN_DIR)/mlp_test: kernels/mlp.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-mlp: $(BIN_DIR)/mlp_test
	./$(BIN_DIR)/mlp_test

$(BIN_DIR)/nexel_color_test: kernels/nexel_color.cu kernels/hash_grid.cu kernels/mlp.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run-nexel-color: $(BIN_DIR)/nexel_color_test
	./$(BIN_DIR)/nexel_color_test

run-intersect: $(BIN_DIR)/intersect_tile
	./$(BIN_DIR)/intersect_tile

$(BIN_DIR)/rasterize_fwd: kernels/rasterize_fwd.cu kernels/projection_2dgs.cu kernels/intersect_tile.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ kernels/rasterize_fwd.cu

$(BIN_DIR)/rasterize_fwd_debug: kernels/rasterize_fwd.cu kernels/projection_2dgs.cu kernels/intersect_tile.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ kernels/rasterize_fwd.cu

run-rasterize: $(BIN_DIR)/rasterize_fwd
	./$(BIN_DIR)/rasterize_fwd

debug-rasterize: $(BIN_DIR)/rasterize_fwd_debug
	cuda-gdb ./$(BIN_DIR)/rasterize_fwd_debug

debug-train: $(BIN_DIR)/train_debug
	cuda-gdb --args ./$(BIN_DIR)/train_debug --data data/360_v2/garden --images images_4

debug: $(BIN_DIR)/camera_projection_debug
	cuda-gdb ./$(BIN_DIR)/camera_projection_debug

debug-quat: $(BIN_DIR)/quat_to_rotmat_debug
	cuda-gdb ./$(BIN_DIR)/quat_to_rotmat_debug

clean:
	rm -rf $(BIN_DIR)
