#pragma once
//
// appearance_adam.cuh
//
// Adam optimiser state for the Nexels appearance parameters (hash grid +
// MLP weights/biases). Kept in a separate header because it depends on
// adam.cu's AdamMomentBuffer / launch_adam_step, which would otherwise
// drag adam.cu into nexel_color.cu's includes and complicate the
// standalone smoke tests.
//
// Lifetime: appearance parameters do not participate in densification,
// so their sizes never change. One AdamMomentBuffer per tensor; one
// step() call advances all five.
//
// The caller (train.cu) #includes adam.cu and then nexel_color.cu before
// this header, so AppearanceField / AdamMomentBuffer / launch_adam_step
// are all visible at the point this is parsed.

struct AppearanceAdamConfig {
    float beta1           = 0.9f;
    float beta2           = 0.999f;
    float eps             = 1e-15f;
    bool  bias_correction = true;

    float lr_grid         = 1e-2f;   // hash-grid features
    float lr_mlp_weights  = 1e-3f;   // W1, W2
    float lr_mlp_biases   = 1e-3f;   // b1, b2
};

class AppearanceAdam {
public:
    AppearanceAdam() = default;

    AppearanceAdam(const AppearanceAdam&)            = delete;
    AppearanceAdam& operator=(const AppearanceAdam&) = delete;

    void allocate(const AppearanceField& af) {
        m_grid.allocate((int)((size_t)af.L * af.T * af.F));
        m_W1.allocate(af.D_hid * af.D_in);
        m_b1.allocate(af.D_hid);
        m_W2.allocate(af.D_out * af.D_hid);
        m_b2.allocate(af.D_out);
        step_count_ = 0;
    }

    void step(AppearanceField& af,
              const AppearanceGrads& g,
              const AppearanceAdamConfig& cfg) {
        step_count_++;
        AdamConfig adam_cfg;
        adam_cfg.beta1           = cfg.beta1;
        adam_cfg.beta2           = cfg.beta2;
        adam_cfg.eps             = cfg.eps;
        adam_cfg.bias_correction = cfg.bias_correction;

        // Each tensor is treated as a flat 1-D parameter group (D=1).
        launch_adam_step(af.grid, g.grad_grid,
                         m_grid.exp_avg, m_grid.exp_avg_sq,
                         m_grid.numel, 1,
                         cfg.lr_grid, adam_cfg, step_count_);
        launch_adam_step(af.W1, g.grad_W1,
                         m_W1.exp_avg, m_W1.exp_avg_sq,
                         m_W1.numel, 1,
                         cfg.lr_mlp_weights, adam_cfg, step_count_);
        launch_adam_step(af.b1, g.grad_b1,
                         m_b1.exp_avg, m_b1.exp_avg_sq,
                         m_b1.numel, 1,
                         cfg.lr_mlp_biases, adam_cfg, step_count_);
        launch_adam_step(af.W2, g.grad_W2,
                         m_W2.exp_avg, m_W2.exp_avg_sq,
                         m_W2.numel, 1,
                         cfg.lr_mlp_weights, adam_cfg, step_count_);
        launch_adam_step(af.b2, g.grad_b2,
                         m_b2.exp_avg, m_b2.exp_avg_sq,
                         m_b2.numel, 1,
                         cfg.lr_mlp_biases, adam_cfg, step_count_);
    }

    int step_count() const { return step_count_; }

private:
    AdamMomentBuffer m_grid;
    AdamMomentBuffer m_W1, m_b1;
    AdamMomentBuffer m_W2, m_b2;
    int step_count_ = 0;
};
