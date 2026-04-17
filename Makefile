NVCC      := nvcc
# -arch=native compiles for the GPU in this machine.
# Swap to e.g. -arch=sm_86 to target a specific architecture (Ampere = sm_80/86).
NVCCFLAGS := -O2 -std=c++17 -arch=sm_89

# -G enables device-side debug info (disables GPU optimisations — debug builds only)
# -g enables host-side debug info
DEBUGFLAGS := -G -g -O0 -std=c++17 -arch=sm_89

BIN_DIR := build
BINS    := $(BIN_DIR)/camera_projection $(BIN_DIR)/quat_to_rotmat $(BIN_DIR)/colmap_reader_test $(BIN_DIR)/splat_data_test $(BIN_DIR)/simple_knn_test $(BIN_DIR)/projection_2dgs $(BIN_DIR)/projection_2dgs_bwd $(BIN_DIR)/gradient_checks $(BIN_DIR)/train $(BIN_DIR)/loss_test $(BIN_DIR)/intersect_tile $(BIN_DIR)/rasterize_fwd $(BIN_DIR)/adam_test

.PHONY: all run-camera run-quat run-knn run-adam run-projection-bwd run-gradcheck run-train-live run-viewer run-viewer-points run-viewer-gsplat debug debug-quat debug-train clean

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

$(BIN_DIR)/gradient_checks: kernels/gradient_checks.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/projection_2dgs_debug: kernels/projection_2dgs.cu kernels/splat_data.cuh kernels/simple_knn.cuh | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

run-projection: $(BIN_DIR)/projection_2dgs
	./$(BIN_DIR)/projection_2dgs

run-projection-bwd: $(BIN_DIR)/projection_2dgs_bwd
	./$(BIN_DIR)/projection_2dgs_bwd

run-gradcheck: $(BIN_DIR)/gradient_checks
	./$(BIN_DIR)/gradient_checks

$(BIN_DIR)/train: train.cu kernels/colmap_reader.hpp kernels/splat_data.cuh kernels/simple_knn.cuh kernels/adam.cu kernels/loss.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu ../LichtFeld-Studio/external/stb_image.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BIN_DIR)/train_debug: train.cu kernels/colmap_reader.hpp kernels/splat_data.cuh kernels/simple_knn.cuh kernels/adam.cu kernels/loss.cu kernels/rasterize_fwd.cu kernels/rasterize_bwd.cu kernels/projection_2dgs.cu kernels/projection_2dgs_bwd.cu kernels/intersect_tile.cu ../LichtFeld-Studio/external/stb_image.h | $(BIN_DIR)
	$(NVCC) $(DEBUGFLAGS) -o $@ $<

run-train: $(BIN_DIR)/train
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4

run-train-live: $(BIN_DIR)/train
	./$(BIN_DIR)/train --data ../data/360_v2/garden --images images_4 --save-ply checkpoints/garden_latest.ply --save-ply-every 50 --densify-start 250 --densify-every 50 --densify-stop 15000

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
