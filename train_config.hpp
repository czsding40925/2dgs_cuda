#pragma once
// Training configuration: CLI flags and their defaults.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct Config {
    std::string data_dir   = "";
    std::string images     = "images";
    std::string ply_path   = "";       // --ply: load trained PLY instead of COLMAP point cloud
    std::string render_out = "";       // --render: render one frame to this PNG and exit
    std::string camera_file = "";      // --camera-file: custom camera spec for --render
    float       render_scale = 1.0f;   // --render-scale: downscale render/orbit resolution
    bool        serve_render = false;  // --serve-render: keep splats resident and render stdin camera requests
    std::string serve_output = "viewer_render.png"; // --serve-output: PNG path written by --serve-render
    int         render_cam = 0;        // --cam: which COLMAP camera to render
    int         orbit_frames = 0;      // --orbit N: render N-frame orbit sequence
    std::string orbit_out  = "orbit";  // --orbit-out: output filename prefix (default "orbit")
    int         orbit_focus_cam = -1;  // --orbit-focus-cam: derive orbit target/start angle from this camera
    float       orbit_radius_scale = 1.0f; // --orbit-radius-scale: tighten or widen orbit radius
    float       orbit_height_scale = 1.0f; // --orbit-height-scale: lower/raise orbit along world-up
    int         iters      = 30000;
    int         sh_degree  = 3;
    int         log_every  = 50;
    int         preview_every = 0;         // save training preview every N iters (0 = off)
    std::string preview_out = "previews/iter"; // output prefix for training previews
    std::string save_ply   = "";           // write latest checkpoint PLY to this path
    int         save_ply_every = 0;        // 0 = only save final checkpoint
    int         densify_every = 100;       // 0 = off
    int         densify_start = 500;
    int         densify_stop  = 15000;
    int         opacity_reset_every = 3000;
    float       dist_lambda = 1e-2f;        // 2DGS depth distortion regularizer
    int         dist_start_iter = 3000;     // enable distortion loss after this iter
    float       normal_lambda = 5e-2f;      // 2DGS normal consistency regularizer
    int         normal_start_iter = 7000;   // enable normal loss after this iter
    float       depth_ratio = 0.f;          // 0 = expected depth (current default),
                                            // 1 = median depth (2DGS paper default).
                                            // Mixes the two as the surface-depth
                                            // input to the normal-from-depth path.
                                            // Median is non-differentiable: at any
                                            // ratio > 0, gradients into the depth
                                            // accumulator are scaled by (1 - ratio).
    float       densify_grad_thresh = 0.0f;     // 0 = use adaptive mean-grad heuristic (recommended; fixed values don't transfer across scenes)
    float       densify_grad_mult   = 3.0f;     // adaptive threshold = mean_grad * this; ~top 30% of Gaussians by gradient
    float       densify_prune_alpha = 0.005f;   // matches gsplat DefaultStrategy
    float       densify_grow_scale3d = 0.01f;   // normalized by scene scale
    float       densify_prune_scale3d = 0.10f;  // normalized by scene scale
    int         max_gaussians = 2000000;        // hard cap: skip densify once N exceeds this

    // --- M3 profiling controls (see notes/perf_plan.md) ---
    int         profile_start_iter = 0;     // 0 = disabled. First iter at which kernel-bucket timers accumulate.
    int         profile_iters      = 0;     // 0 = disabled. Number of iters to accumulate after profile_start_iter.
    std::string profile_csv        = "";    // path to append per-bucket CSV row (per-bucket row).
    std::string profile_tag        = "run"; // string written into the "tag" column of profile_csv.
    bool        profile_exit       = false; // exit training after the profile window closes.

    // --- M4 distributed-training controls (see TODO.md §M4) ---
    // Only meaningful in the train_mpi binary (built with -DUSE_MPI).
    // The single-process train binary ignores them.
    uint64_t    seed              = 42;     // master RNG seed; broadcast from rank 0 if MPI is on.
    int         cameras_per_rank  = 1;      // per-iter cameras processed by each rank; effective batch = world_size * cameras_per_rank.
    bool        mpi_host_stage    = false;  // force host-staged allreduce path (instead of CUDA-aware). For comparison numbers.

    // --- M5 Nexels appearance pipeline (see TODO.md §"Phase 3 — Nexels") ---
    // When use_nexel is true the trainer swaps the SH-based per-Gaussian
    // colour computation for a shared hash grid + small MLP queried at
    // each Gaussian's normalised mean position. SH parameters remain
    // allocated but receive no gradient.
    bool        use_nexel             = false;
    int         hash_levels           = 16;   // L
    int         hash_log_table_size   = 19;   // T = 2^log_table_size (=512K cells per level)
    int         hash_features         = 2;    // F per level
    float       hash_min_res          = 16.f; // coarsest voxel resolution (in unit-cube space)
    float       hash_max_res          = 1024.f;// finest voxel resolution
    int         mlp_hidden            = 64;   // D_hid
    float       lr_grid               = 1e-2f;// hash-grid LR (Nexels default)
    float       lr_mlp                = 1e-3f;// MLP weights + biases LR
    uint64_t    nexel_init_seed       = 42;   // RNG seed for AppearanceField::init_random
    // View-dependent path: MLP outputs 48 SH coefficients (degree 3,
    // sh0 + shN) and a packed-SH eval kernel evaluates them per
    // Gaussian using the camera position. Closes most of the SH-vs-
    // Nexels quality gap without rasterizer surgery.
    bool        nexel_view_dep        = false;
    // Disable warp-dedup in the hash-grid backward (uses straight
    // atomicAdd instead). For A/B benchmarking the two kernel variants.
    bool        hash_bwd_naive        = false;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--data") == 0 && i+1 < argc)
            cfg.data_dir = argv[++i];
        else if (std::strcmp(argv[i], "--images") == 0 && i+1 < argc)
            cfg.images = argv[++i];
        else if (std::strcmp(argv[i], "--ply") == 0 && i+1 < argc)
            cfg.ply_path = argv[++i];
        else if (std::strcmp(argv[i], "--render") == 0 && i+1 < argc)
            cfg.render_out = argv[++i];
        else if (std::strcmp(argv[i], "--camera-file") == 0 && i+1 < argc)
            cfg.camera_file = argv[++i];
        else if (std::strcmp(argv[i], "--render-scale") == 0 && i+1 < argc)
            cfg.render_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--serve-render") == 0)
            cfg.serve_render = true;
        else if (std::strcmp(argv[i], "--serve-output") == 0 && i+1 < argc)
            cfg.serve_output = argv[++i];
        else if (std::strcmp(argv[i], "--cam") == 0 && i+1 < argc)
            cfg.render_cam = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--orbit") == 0 && i+1 < argc)
            cfg.orbit_frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--orbit-out") == 0 && i+1 < argc)
            cfg.orbit_out = argv[++i];
        else if (std::strcmp(argv[i], "--orbit-focus-cam") == 0 && i+1 < argc)
            cfg.orbit_focus_cam = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--orbit-radius-scale") == 0 && i+1 < argc)
            cfg.orbit_radius_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--orbit-height-scale") == 0 && i+1 < argc)
            cfg.orbit_height_scale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i+1 < argc)
            cfg.iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sh-degree") == 0 && i+1 < argc)
            cfg.sh_degree = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--log-every") == 0 && i+1 < argc)
            cfg.log_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--preview-every") == 0 && i+1 < argc)
            cfg.preview_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--preview-out") == 0 && i+1 < argc)
            cfg.preview_out = argv[++i];
        else if (std::strcmp(argv[i], "--save-ply") == 0 && i+1 < argc)
            cfg.save_ply = argv[++i];
        else if (std::strcmp(argv[i], "--save-ply-every") == 0 && i+1 < argc)
            cfg.save_ply_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-every") == 0 && i+1 < argc)
            cfg.densify_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-start") == 0 && i+1 < argc)
            cfg.densify_start = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-stop") == 0 && i+1 < argc)
            cfg.densify_stop = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--opacity-reset-every") == 0 && i+1 < argc)
            cfg.opacity_reset_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dist-lambda") == 0 && i+1 < argc)
            cfg.dist_lambda = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dist-start-iter") == 0 && i+1 < argc)
            cfg.dist_start_iter = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--normal-lambda") == 0 && i+1 < argc)
            cfg.normal_lambda = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--normal-start-iter") == 0 && i+1 < argc)
            cfg.normal_start_iter = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--depth-ratio") == 0 && i+1 < argc)
            cfg.depth_ratio = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-grad-thresh") == 0 && i+1 < argc)
            cfg.densify_grad_thresh = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-grad-mult") == 0 && i+1 < argc)
            cfg.densify_grad_mult = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-prune-alpha") == 0 && i+1 < argc)
            cfg.densify_prune_alpha = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-grow-scale3d") == 0 && i+1 < argc)
            cfg.densify_grow_scale3d = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--densify-prune-scale3d") == 0 && i+1 < argc)
            cfg.densify_prune_scale3d = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--max-gaussians") == 0 && i+1 < argc)
            cfg.max_gaussians = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--profile-start-iter") == 0 && i+1 < argc)
            cfg.profile_start_iter = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--profile-iters") == 0 && i+1 < argc)
            cfg.profile_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--profile-csv") == 0 && i+1 < argc)
            cfg.profile_csv = argv[++i];
        else if (std::strcmp(argv[i], "--profile-tag") == 0 && i+1 < argc)
            cfg.profile_tag = argv[++i];
        else if (std::strcmp(argv[i], "--profile-exit") == 0)
            cfg.profile_exit = true;
        else if (std::strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            cfg.seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--cameras-per-rank") == 0 && i+1 < argc)
            cfg.cameras_per_rank = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--mpi-host-stage") == 0)
            cfg.mpi_host_stage = true;
        else if (std::strcmp(argv[i], "--use-nexel") == 0)
            cfg.use_nexel = true;
        else if (std::strcmp(argv[i], "--hash-levels") == 0 && i+1 < argc)
            cfg.hash_levels = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--hash-log-table-size") == 0 && i+1 < argc)
            cfg.hash_log_table_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--hash-features") == 0 && i+1 < argc)
            cfg.hash_features = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--hash-min-res") == 0 && i+1 < argc)
            cfg.hash_min_res = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--hash-max-res") == 0 && i+1 < argc)
            cfg.hash_max_res = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--mlp-hidden") == 0 && i+1 < argc)
            cfg.mlp_hidden = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--lr-grid") == 0 && i+1 < argc)
            cfg.lr_grid = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--lr-mlp") == 0 && i+1 < argc)
            cfg.lr_mlp = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--nexel-init-seed") == 0 && i+1 < argc)
            cfg.nexel_init_seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--nexel-view-dep") == 0)
            cfg.nexel_view_dep = true;
        else if (std::strcmp(argv[i], "--hash-bwd-naive") == 0)
            cfg.hash_bwd_naive = true;
        else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: train [--data <dir>] [--images images] [--ply file.ply]\n");
            fprintf(stderr, "             [--render out.png] [--camera-file spec.txt] [--cam N] [--render-scale S]\n");
            fprintf(stderr, "             [--serve-render] [--serve-output out.png]\n");
            fprintf(stderr, "             [--orbit N] [--orbit-out prefix] [--orbit-focus-cam N]\n");
            fprintf(stderr, "             [--orbit-radius-scale R] [--orbit-height-scale H]\n");
            fprintf(stderr, "             [--iters N] [--sh-degree 0-3] [--log-every N]\n");
            fprintf(stderr, "             [--preview-every N] [--preview-out prefix]\n");
            fprintf(stderr, "             [--save-ply path] [--save-ply-every N]\n");
            fprintf(stderr, "             [--densify-every N] [--densify-start N] [--densify-stop N]\n");
            fprintf(stderr, "             [--dist-lambda X] [--dist-start-iter N] [--normal-lambda X] [--normal-start-iter N]\n");
            fprintf(stderr, "             [--opacity-reset-every N] [--densify-grad-thresh X] [--densify-grad-mult X]\n");
            fprintf(stderr, "             [--densify-prune-alpha X] [--densify-grow-scale3d X] [--densify-prune-scale3d X]\n");
            fprintf(stderr, "             [--max-gaussians N]\n");
            fprintf(stderr, "             [--profile-start-iter K] [--profile-iters N] [--profile-csv path]\n");
            fprintf(stderr, "             [--profile-tag s] [--profile-exit]\n");
            fprintf(stderr, "             [--seed N] [--cameras-per-rank K] [--mpi-host-stage]\n");
            fprintf(stderr, "             [--use-nexel] [--hash-levels L] [--hash-log-table-size K] [--hash-features F]\n");
            fprintf(stderr, "             [--hash-min-res X] [--hash-max-res X] [--mlp-hidden N]\n");
            fprintf(stderr, "             [--lr-grid X] [--lr-mlp X] [--nexel-init-seed N] [--nexel-view-dep]\n");
            exit(1);
        }
    }
    if (cfg.data_dir.empty() && cfg.ply_path.empty()) {
        fprintf(stderr, "Error: provide --data <dir> or --ply <file>\n");
        exit(1);
    }
    if (cfg.render_scale <= 0.f) {
        fprintf(stderr, "Error: --render-scale must be > 0\n");
        exit(1);
    }
    if (cfg.orbit_radius_scale <= 0.f) {
        fprintf(stderr, "Error: --orbit-radius-scale must be > 0\n");
        exit(1);
    }
    if (cfg.orbit_height_scale < 0.f) {
        fprintf(stderr, "Error: --orbit-height-scale must be >= 0\n");
        exit(1);
    }
    if (cfg.log_every <= 0) {
        fprintf(stderr, "Error: --log-every must be > 0\n");
        exit(1);
    }
    if (cfg.preview_every < 0) {
        fprintf(stderr, "Error: --preview-every must be >= 0\n");
        exit(1);
    }
    if (cfg.save_ply_every < 0) {
        fprintf(stderr, "Error: --save-ply-every must be >= 0\n");
        exit(1);
    }
    if (cfg.save_ply_every > 0 && cfg.save_ply.empty()) {
        fprintf(stderr, "Error: --save-ply-every requires --save-ply <path>\n");
        exit(1);
    }
    if (cfg.densify_every < 0) {
        fprintf(stderr, "Error: --densify-every must be >= 0\n");
        exit(1);
    }
    if (cfg.opacity_reset_every < 0) {
        fprintf(stderr, "Error: --opacity-reset-every must be >= 0\n");
        exit(1);
    }
    if (cfg.densify_start < 0 || cfg.densify_stop < 0) {
        fprintf(stderr, "Error: densification iteration bounds must be >= 0\n");
        exit(1);
    }
    if (cfg.dist_lambda < 0.f || cfg.normal_lambda < 0.f) {
        fprintf(stderr, "Error: geometry regularization weights must be >= 0\n");
        exit(1);
    }
    if (cfg.depth_ratio < 0.f || cfg.depth_ratio > 1.f) {
        fprintf(stderr, "Error: --depth-ratio must be in [0, 1]\n");
        exit(1);
    }
    if (cfg.dist_start_iter < 0 || cfg.normal_start_iter < 0) {
        fprintf(stderr, "Error: geometry regularization start iterations must be >= 0\n");
        exit(1);
    }
    if (cfg.densify_every > 0 && cfg.densify_stop > 0 && cfg.densify_stop < cfg.densify_start) {
        fprintf(stderr, "Error: --densify-stop must be >= --densify-start\n");
        exit(1);
    }
    if (cfg.densify_grad_thresh < 0.f || cfg.densify_prune_alpha <= 0.f || cfg.densify_prune_alpha >= 1.f ||
        cfg.densify_grow_scale3d <= 0.f || cfg.densify_prune_scale3d <= 0.f) {
        fprintf(stderr, "Error: densification thresholds must be positive, and prune alpha must be in (0,1)\n");
        exit(1);
    }
    if (cfg.profile_start_iter < 0 || cfg.profile_iters < 0) {
        fprintf(stderr, "Error: --profile-start-iter / --profile-iters must be >= 0\n");
        exit(1);
    }
    if (cfg.cameras_per_rank < 1) {
        fprintf(stderr, "Error: --cameras-per-rank must be >= 1\n");
        exit(1);
    }
    if (cfg.use_nexel) {
        if (cfg.hash_levels < 1 || cfg.hash_levels > 32) {
            fprintf(stderr, "Error: --hash-levels must be in [1, 32]\n"); exit(1);
        }
        if (cfg.hash_log_table_size < 4 || cfg.hash_log_table_size > 24) {
            fprintf(stderr, "Error: --hash-log-table-size must be in [4, 24]\n"); exit(1);
        }
        if (cfg.hash_features < 1 || cfg.hash_features > 8) {
            fprintf(stderr, "Error: --hash-features must be in [1, 8]\n"); exit(1);
        }
        if (cfg.hash_min_res <= 0.f || cfg.hash_max_res <= cfg.hash_min_res) {
            fprintf(stderr, "Error: --hash-min-res < --hash-max-res, both > 0\n"); exit(1);
        }
        if (cfg.mlp_hidden < 1 || cfg.mlp_hidden > 256) {
            fprintf(stderr, "Error: --mlp-hidden must be in [1, 256]\n"); exit(1);
        }
        if (cfg.lr_grid <= 0.f || cfg.lr_mlp <= 0.f) {
            fprintf(stderr, "Error: --lr-grid / --lr-mlp must be > 0\n"); exit(1);
        }
    }
    if (!cfg.camera_file.empty() && cfg.render_out.empty()) {
        fprintf(stderr, "Error: --camera-file requires --render <out.png>\n");
        exit(1);
    }
    if (cfg.serve_render && cfg.ply_path.empty()) {
        fprintf(stderr, "Error: --serve-render requires --ply <file>\n");
        exit(1);
    }
    if (cfg.serve_render && cfg.serve_output.empty()) {
        fprintf(stderr, "Error: --serve-output must not be empty\n");
        exit(1);
    }
    const bool needs_scene =
        (!cfg.render_out.empty() && cfg.camera_file.empty()) ||
        (cfg.orbit_frames > 0) ||
        (cfg.render_out.empty() && cfg.orbit_frames == 0 && !cfg.serve_render);
    if (needs_scene && cfg.data_dir.empty()) {
        fprintf(stderr, "Error: --data <dir> is required for training, scene-camera renders, and orbit renders\n");
        exit(1);
    }
    return cfg;
}
