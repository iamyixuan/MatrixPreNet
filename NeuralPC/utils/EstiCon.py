import torch
import torch.linalg.norm as norm


class EstiCon:
    def __init__(self, c1, c2, c3, c4, c1_prime):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c1_prime = c1_prime
        self.max_iter = 1000

    def max_sigma(self, linearOpt):
        pass

    def min_sigma(self, linearOpt):

        sig_max, v_max = self.max_sigma(linearOpt)
        sig_min, v_min = sig_max, v_max

        x = self.draw_v(v_max.shape)
        tau = torch.erfinv(self.c2) / norm(x, 2)
        x_star = x / norm(x, 2)

        b = linearOpt(x_star)
        beta = norm(b, 2)
        u = b / beta

        v = linearOpt(u)
        alpha = norm(v, 2)
        v = v / alpha

        w = v
        x = torch.zeros_like(x)

        phi_bar = beta
        pho_bar = alpha

        theta_tminus = 0
        R_tt = torch.zeros(x_star.shape - 1)
        for t in range(1, self.max_iter):
            u = linearOpt(v) - alpha * u
            beta = norm(u, 2)
            u = u / beta
            v = linearOpt(u) - beta * v
            alpha = norm(v, 2)
            v = v / alpha
            pho = norm(pho_bar * beta, 2)  # might be wrong
            c = pho_bar / pho
            s = beta / pho
            theta = s * alpha

            if t == 1:
                theta_tminus = theta

            pho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar = s * phi_bar
            x = x + (phi / pho) * w
            w = v - (theta / pho) * w
            R_tt = pho
            if t > 1:
                R_tm1_t = theta_tminus
                theta_tminus = theta
            d = x_star - x

            if d == 0:
                sig_min = sig_max
                v_min = v_max
                break

            Ad_norm = norm(linearOpt(d), 2)
            d_norm = norm(d, 2)

            if Ad_norm <= sig_min * d_norm:
                sig_min = Ad_norm / d_norm
                v_min = d

            if sig_max / sig_min >= self.c4:
                self.c1 = self.c1_prime

        return

    def draw_v(self, shape):
        # Draw a random complex vector v
        v = torch.randn(shape, dtype=torch.complex64)
        return v
