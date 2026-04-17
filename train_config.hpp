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
    float       densify_grad_thresh = 0.0f;     // 0 = use adaptive mean-grad heuristic (recommended; fixed values don't transfer across scenes)
    float       densify_grad_mult   = 3.0f;     // adaptive threshold = mean_grad * this; ~top 30% of Gaussians by gradient
    float       densify_prune_alpha = 0.005f;   // matches gsplat DefaultStrategy
    float       densify_grow_scale3d = 0.01f;   // normalized by scene scale
    float       densify_prune_scale3d = 0.10f;  // normalized by scene scale
    int         max_gaussians = 2000000;        // hard cap: skip densify once N exceeds this
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
