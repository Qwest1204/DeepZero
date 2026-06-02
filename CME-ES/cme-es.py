import math
import torch
import numpy as np

class CMAES:
    def __init__(self, x0: torch.Tensor, sigma0: float,
                 popsize: int = None, mu: int = None,
                 device: str = 'cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.N = x0.shape[0]
        
        self.lamb = popsize if popsize else 4 + int(3 * math.log(self.N))
        self.mu = mu if mu else self.lamb // 2
        
        w = torch.tensor([math.log(self.mu + 0.5) - math.log(i + 1)
                          for i in range(self.mu)], device=device, dtype=dtype)
        w /= w.sum()
        self.w = w
        self.mueff = 1.0 / (w ** 2).sum()
        
        self.cc = (4.0 + self.mueff / self.N) / (self.N + 4.0 + 2.0 * self.mueff / self.N)
        self.cs = (self.mueff + 2.0) / (self.N + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.N + 2.0) ** 2 + self.mueff))
        self.damps = (1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (self.N + 1.0)) - 1.0) + self.cs)
        self.chiN = math.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N ** 2))
        
        self.mean = x0.to(device=device, dtype=dtype)
        self.sigma = torch.tensor(sigma0, device=device, dtype=dtype)
        self.C = torch.eye(self.N, device=device, dtype=dtype)
        self.pc = torch.zeros(self.N, device=device, dtype=dtype)
        self.ps = torch.zeros(self.N, device=device, dtype=dtype)
        
        self._eigen_decomposition()
        
        self.z = None
        self.y = None
        self.x = None
        
        self.generation = 0
        self.best_x = self.mean.clone()
        self.best_f = float('inf')
    
    def _eigen_decomposition(self):
        D, B = torch.linalg.eigh(self.C)
        D = torch.clamp(D, min=1e-10)
        self.B = B
        self.D = D
        self.sqrtD = torch.sqrt(D)
        self.invsqrtD = 1.0 / self.sqrtD
    
    def ask(self) -> torch.Tensor:
        """Возвращает популяцию-кандидатов (lambda, N)."""
        self.z = torch.randn(self.lamb, self.N, device=self.device, dtype=self.dtype)
        self.y = (self.z * self.sqrtD) @ self.B.T
        self.x = self.mean + self.sigma * self.y
        return self.x
    
    def tell(self, fitnesses: torch.Tensor):
        idx = torch.argsort(fitnesses)
        fitnesses = fitnesses[idx]

        # запоминаем лучшее значение текущего поколения
        current_best_f = fitnesses[0].item()
        if current_best_f < self.best_f:
            self.best_f = current_best_f
            self.best_x = self.x[idx[0]].clone()
        
        # Упорядочиваем z, y, x
        z = self.z[idx]
        y = self.y[idx]
        x = self.x[idx]
        
        # Взвешенная рекомбинация (только mu лучших)
        yw = (self.w.unsqueeze(1) * y[:self.mu]).sum(dim=0)   # y_w
        self.mean = self.mean + self.sigma * yw
        
        # Обратный квадратный корень из C, умноженный на yw
        # C^{-1/2} * yw = B * diag(1/sqrt(D)) * B^T * yw
        Bt_yw = self.B.T @ yw
        inv_sqrtC_yw = self.B @ (Bt_yw * self.invsqrtD)
        
        # Обновление эволюционного пути для шага sigma
        self.ps = ((1 - self.cs) * self.ps +
                   math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrtC_yw)
        
        # Обновление sigma
        ps_norm = torch.norm(self.ps)
        self.sigma *= torch.exp(self.cs / self.damps * (ps_norm / self.chiN - 1.0)).item()
        
        # Пороговое условие для hsig
        hsig = (ps_norm / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
                < (1.4 + 2.0 / (self.N + 1.0)) * self.chiN)
        hsig = float(hsig)  # 1.0 или 0.0
        
        # Обновление эволюционного пути для ковариационной матрицы
        self.pc = ((1 - self.cc) * self.pc +
                   hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw)
        
        # Ранговое обновление ковариационной матрицы
        # C = (1 - c1 - cmu) * C  +  c1 * (pc*pc^T + (1-hsig)*cc*(2-cc)*C)
        #      + cmu * sum_{i=1}^{mu} w_i * y_i * y_i^T
        pc_outer = torch.outer(self.pc, self.pc)
        rank_mu = torch.zeros_like(self.C)
        for i in range(self.mu):
            rank_mu += self.w[i] * torch.outer(y[i], y[i])
        
        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (pc_outer + (1 - hsig) * self.cc * (2 - self.cc) * self.C) +
                  self.cmu * rank_mu)
        
        # Обновляем собственное разложение
        self._eigen_decomposition()
        
        self.generation += 1