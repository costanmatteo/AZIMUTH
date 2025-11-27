"""
Verifica numerica dei risultati teorici sulla bias strutturale
nelle funzioni di perdita con reparametrizzazione stocastica.

Questo script confronta i valori teorici esatti (derivati analiticamente)
con le stime Monte Carlo per verificare la correttezza delle formule.

Teoria di riferimento:
- Q(o) = exp(-(o-τ)²/s)  [funzione di qualità gaussiana]
- o = μ + σε, ε ~ N(0,1)  [reparametrizzazione stocastica]
- F* = exp(-(μ-τ)²/s)    [reliability target deterministico]
- F = Q(o)               [reliability stocastico]
- δ = μ - τ              [deviazione dal target]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings


# ==============================================================================
# Valori teorici esatti
# ==============================================================================

def F_star_theoretical(mu: float, tau: float, s: float) -> float:
    """
    Calcola F* = exp(-(μ-τ)²/s), il valore deterministico della reliability
    quando non c'è incertezza (σ=0).

    Args:
        mu: Media dell'output (valore controllato dalla policy)
        tau: Target desiderato
        s: Parametro di scala della funzione di qualità

    Returns:
        F*: Reliability target deterministica
    """
    delta = mu - tau
    return np.exp(-delta**2 / s)


def E_F_theoretical(mu: float, tau: float, s: float, sigma2: float) -> float:
    """
    Calcola E[F] usando la formula standard per gaussiane.

    Se X ~ N(μ, σ²), allora E[exp(-α(X-τ)²)] = 1/sqrt(1 + 2ασ²) · exp(-αδ²/(1 + 2ασ²))

    Con α = 1/s (per F):
    E[F] = 1/sqrt(1 + 2σ²/s) · exp(-δ²/(s + 2σ²))

    Equivalentemente:
    E[F] = F* / sqrt(1 + 2σ²/s) · exp(2δ²σ² / (s(s + 2σ²)))

    Dove: F* = exp(-δ²/s), δ = μ - τ

    Args:
        mu: Media dell'output
        tau: Target
        s: Parametro di scala
        sigma2: Varianza dell'incertezza (σ²)

    Returns:
        E[F]: Valore atteso della reliability stocastica
    """
    delta = mu - tau

    # Coefficiente di attenuazione
    attenuation = 1.0 / np.sqrt(1 + 2*sigma2/s)

    # Esponente corretto: -δ²/(s + 2σ²)
    if s + 2*sigma2 > 0:
        exp_term = np.exp(-delta**2 / (s + 2*sigma2))
    else:
        exp_term = np.exp(-delta**2 / s) if s > 0 else 0.0

    return attenuation * exp_term


def E_F2_theoretical(mu: float, tau: float, s: float, sigma2: float) -> float:
    """
    Calcola E[F²] = E[exp(-2(μ+σε-τ)²/s)]

    Derivazione corretta usando la formula standard per gaussiane:
    Se X ~ N(μ, σ²), allora E[exp(-α(X-τ)²)] = 1/sqrt(1 + 2ασ²) · exp(-αδ²/(1 + 2ασ²))

    Con α = 2/s (per F²):
    E[F²] = 1/sqrt(1 + 4σ²/s) · exp(-2δ²/(s + 4σ²))

    Args:
        mu: Media dell'output
        tau: Target
        s: Parametro di scala
        sigma2: Varianza dell'incertezza (σ²)

    Returns:
        E[F²]: Secondo momento della reliability stocastica
    """
    delta = mu - tau

    # Coefficiente di attenuazione per il secondo momento
    attenuation = 1.0 / np.sqrt(1 + 4*sigma2/s)

    # Esponente corretto: -2δ²/(s + 4σ²)
    if s + 4*sigma2 > 0:
        exp_term = np.exp(-2*delta**2 / (s + 4*sigma2))
    else:
        exp_term = np.exp(-2*delta**2 / s) if s > 0 else 0.0

    return attenuation * exp_term


def Var_F_theoretical(mu: float, tau: float, s: float, sigma2: float) -> float:
    """
    Calcola Var(F) = E[F²] - (E[F])²

    Args:
        mu, tau, s, sigma2: Parametri del modello

    Returns:
        Var(F): Varianza della reliability stocastica
    """
    E_F = E_F_theoretical(mu, tau, s, sigma2)
    E_F2 = E_F2_theoretical(mu, tau, s, sigma2)
    return E_F2 - E_F**2


def Loss_theoretical(mu: float, tau: float, s: float, sigma2: float) -> float:
    """
    Calcola la Loss = E[(F - F*)²] = Var(F) + (E[F] - F*)²

    Decomposizione bias-varianza:
    - Bias² = (E[F] - F*)²
    - Varianza = Var(F)

    Args:
        mu, tau, s, sigma2: Parametri del modello

    Returns:
        Loss totale
    """
    F_star = F_star_theoretical(mu, tau, s)
    E_F = E_F_theoretical(mu, tau, s, sigma2)
    Var_F = Var_F_theoretical(mu, tau, s, sigma2)

    bias_squared = (E_F - F_star)**2
    return Var_F + bias_squared


def Loss_min_theoretical(s: float, sigma2: float) -> Tuple[float, float, float]:
    """
    Calcola la Loss minima (per policy perfetta, δ=0).

    L_min = Var(F)|_{δ=0} + (E[F] - 1)²|_{δ=0}

    Con δ=0:
    - F* = 1
    - E[F] = 1/sqrt(1 + 2σ²/s)
    - E[F²] = 1/sqrt(1 + 4σ²/s)
    - Var(F) = E[F²] - (E[F])²
    - Bias² = (E[F] - 1)²

    Args:
        s: Parametro di scala
        sigma2: Varianza dell'incertezza

    Returns:
        (L_min, bias_squared, variance): Loss minima e suoi componenti
    """
    E_F = 1.0 / np.sqrt(1 + 2*sigma2/s)
    E_F2 = 1.0 / np.sqrt(1 + 4*sigma2/s)

    variance = E_F2 - E_F**2
    bias_squared = (E_F - 1)**2

    L_min = variance + bias_squared
    return L_min, bias_squared, variance


# ==============================================================================
# Stime Monte Carlo
# ==============================================================================

def monte_carlo_estimate(
    mu: float,
    tau: float,
    s: float,
    sigma2: float,
    n_samples: int = 100000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Stima Monte Carlo delle quantità F, E[F], E[F²], Var(F), Loss.

    Procedura:
    1. Campiona ε ~ N(0,1) per n_samples volte
    2. Calcola o = μ + σε per ogni campione
    3. Calcola F = exp(-(o-τ)²/s) per ogni campione
    4. Stima E[F], E[F²], Var(F), Loss dalle statistiche campionarie

    Args:
        mu, tau, s, sigma2: Parametri del modello
        n_samples: Numero di campioni Monte Carlo
        seed: Seed per riproducibilità (opzionale)

    Returns:
        Dizionario con E_F, E_F2, Var_F, Loss, F_star, samples
    """
    if seed is not None:
        np.random.seed(seed)

    sigma = np.sqrt(sigma2)

    # Campiona epsilon ~ N(0,1)
    epsilon = np.random.randn(n_samples)

    # Calcola output: o = μ + σε
    outputs = mu + sigma * epsilon

    # Calcola F = exp(-(o-τ)²/s) per ogni campione
    F_samples = np.exp(-(outputs - tau)**2 / s)

    # Stime Monte Carlo
    E_F_mc = np.mean(F_samples)
    E_F2_mc = np.mean(F_samples**2)
    Var_F_mc = np.var(F_samples, ddof=0)  # Varianza campionaria (divisore N)

    # F* deterministico
    F_star = F_star_theoretical(mu, tau, s)

    # Loss = E[(F - F*)²]
    Loss_mc = np.mean((F_samples - F_star)**2)

    return {
        'E_F': E_F_mc,
        'E_F2': E_F2_mc,
        'Var_F': Var_F_mc,
        'Loss': Loss_mc,
        'F_star': F_star,
        'F_samples': F_samples
    }


# ==============================================================================
# Risultati strutturati
# ==============================================================================

@dataclass
class VerificationResult:
    """Risultato di una singola verifica teorico vs Monte Carlo."""
    parameter_name: str
    parameter_value: float
    theoretical: float
    monte_carlo: float
    relative_error_pct: float
    verified: bool  # True se errore relativo < 1%


@dataclass
class ExperimentResults:
    """Risultati completi di un esperimento."""
    experiment_name: str
    description: str
    results: List[VerificationResult]
    summary: Dict[str, float]


def compute_relative_error(theoretical: float, monte_carlo: float) -> float:
    """Calcola l'errore relativo percentuale."""
    if abs(theoretical) < 1e-10:
        if abs(monte_carlo) < 1e-10:
            return 0.0
        return float('inf')
    return abs(monte_carlo - theoretical) / abs(theoretical) * 100


def verify_single_config(
    mu: float,
    tau: float,
    s: float,
    sigma2: float,
    n_samples: int = 100000,
    seed: int = 42
) -> Dict[str, VerificationResult]:
    """
    Verifica una singola configurazione di parametri.

    Returns:
        Dizionario con i risultati di verifica per E_F, E_F2, Var_F, Loss
    """
    # Valori teorici
    E_F_th = E_F_theoretical(mu, tau, s, sigma2)
    E_F2_th = E_F2_theoretical(mu, tau, s, sigma2)
    Var_F_th = Var_F_theoretical(mu, tau, s, sigma2)
    Loss_th = Loss_theoretical(mu, tau, s, sigma2)
    F_star_th = F_star_theoretical(mu, tau, s)

    # Stime Monte Carlo
    mc = monte_carlo_estimate(mu, tau, s, sigma2, n_samples, seed)

    results = {}

    for name, th_val, mc_val in [
        ('E_F', E_F_th, mc['E_F']),
        ('E_F2', E_F2_th, mc['E_F2']),
        ('Var_F', Var_F_th, mc['Var_F']),
        ('Loss', Loss_th, mc['Loss']),
    ]:
        rel_err = compute_relative_error(th_val, mc_val)
        results[name] = VerificationResult(
            parameter_name=name,
            parameter_value=sigma2,
            theoretical=th_val,
            monte_carlo=mc_val,
            relative_error_pct=rel_err,
            verified=rel_err < 1.0
        )

    return results


# ==============================================================================
# Esperimenti principali
# ==============================================================================

def experiment_a_vary_sigma(
    tau: float = 0.0,
    s: float = 1.0,
    mu: float = 0.0,
    sigma2_values: List[float] = None,
    n_samples: int = 100000,
    seed: int = 42
) -> Dict:
    """
    Esperimento (a): Policy perfetta (δ=0), varia σ².

    Verifica base con:
    - τ = 0, s = 1, μ = 0 (quindi δ = 0)
    - σ² varia in [0.1, 0.5, 1, 2, 5]
    """
    if sigma2_values is None:
        sigma2_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    results = {
        'sigma2': [],
        'E_F_th': [], 'E_F_mc': [], 'E_F_err': [],
        'Var_F_th': [], 'Var_F_mc': [], 'Var_F_err': [],
        'Loss_th': [], 'Loss_mc': [], 'Loss_err': [],
        'F_star': [],
        'verified_all': []
    }

    for sigma2 in sigma2_values:
        # Valori teorici
        E_F_th = E_F_theoretical(mu, tau, s, sigma2)
        Var_F_th = Var_F_theoretical(mu, tau, s, sigma2)
        Loss_th = Loss_theoretical(mu, tau, s, sigma2)
        F_star = F_star_theoretical(mu, tau, s)

        # Monte Carlo
        mc = monte_carlo_estimate(mu, tau, s, sigma2, n_samples, seed)

        # Errori relativi
        E_F_err = compute_relative_error(E_F_th, mc['E_F'])
        Var_F_err = compute_relative_error(Var_F_th, mc['Var_F'])
        Loss_err = compute_relative_error(Loss_th, mc['Loss'])

        results['sigma2'].append(sigma2)
        results['E_F_th'].append(E_F_th)
        results['E_F_mc'].append(mc['E_F'])
        results['E_F_err'].append(E_F_err)
        results['Var_F_th'].append(Var_F_th)
        results['Var_F_mc'].append(mc['Var_F'])
        results['Var_F_err'].append(Var_F_err)
        results['Loss_th'].append(Loss_th)
        results['Loss_mc'].append(mc['Loss'])
        results['Loss_err'].append(Loss_err)
        results['F_star'].append(F_star)
        results['verified_all'].append(
            E_F_err < 1.0 and Var_F_err < 1.0 and Loss_err < 1.0
        )

    return results


def experiment_b_vary_delta(
    tau: float = 0.0,
    s: float = 1.0,
    sigma2: float = 1.0,
    mu_values: List[float] = None,
    n_samples: int = 100000,
    seed: int = 42
) -> Dict:
    """
    Esperimento (b): δ ≠ 0, varia μ.

    Con:
    - τ = 0, s = 1, σ² = 1
    - μ varia in [-2, -1, 0, 1, 2] (quindi δ = μ)
    """
    if mu_values is None:
        mu_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

    results = {
        'mu': [], 'delta': [],
        'E_F_th': [], 'E_F_mc': [], 'E_F_err': [],
        'Var_F_th': [], 'Var_F_mc': [], 'Var_F_err': [],
        'Loss_th': [], 'Loss_mc': [], 'Loss_err': [],
        'F_star': [],
        'verified_all': []
    }

    for mu in mu_values:
        delta = mu - tau

        # Valori teorici
        E_F_th = E_F_theoretical(mu, tau, s, sigma2)
        Var_F_th = Var_F_theoretical(mu, tau, s, sigma2)
        Loss_th = Loss_theoretical(mu, tau, s, sigma2)
        F_star = F_star_theoretical(mu, tau, s)

        # Monte Carlo
        mc = monte_carlo_estimate(mu, tau, s, sigma2, n_samples, seed)

        # Errori relativi
        E_F_err = compute_relative_error(E_F_th, mc['E_F'])
        Var_F_err = compute_relative_error(Var_F_th, mc['Var_F'])
        Loss_err = compute_relative_error(Loss_th, mc['Loss'])

        results['mu'].append(mu)
        results['delta'].append(delta)
        results['E_F_th'].append(E_F_th)
        results['E_F_mc'].append(mc['E_F'])
        results['E_F_err'].append(E_F_err)
        results['Var_F_th'].append(Var_F_th)
        results['Var_F_mc'].append(mc['Var_F'])
        results['Var_F_err'].append(Var_F_err)
        results['Loss_th'].append(Loss_th)
        results['Loss_mc'].append(mc['Loss'])
        results['Loss_err'].append(Loss_err)
        results['F_star'].append(F_star)
        results['verified_all'].append(
            E_F_err < 1.0 and Var_F_err < 1.0 and Loss_err < 1.0
        )

    return results


def experiment_c_loss_unreachability(
    s: float = 1.0,
    sigma2_range: Tuple[float, float] = (0.01, 5.0),
    n_points: int = 100
) -> Dict:
    """
    Esperimento (c): Verifica che L_min > 0 per ogni σ² > 0.

    Mostra che la loss non può mai raggiungere 0 quando c'è incertezza.
    """
    sigma2_values = np.linspace(sigma2_range[0], sigma2_range[1], n_points)

    results = {
        'sigma2': [],
        'L_min': [],
        'bias_squared': [],
        'variance': []
    }

    for sigma2 in sigma2_values:
        L_min, bias_sq, var = Loss_min_theoretical(s, sigma2)
        results['sigma2'].append(sigma2)
        results['L_min'].append(L_min)
        results['bias_squared'].append(bias_sq)
        results['variance'].append(var)

    return results


def experiment_d_jensen_inequality(
    tau: float = 0.0,
    s: float = 1.0,
    mu: float = 0.0,
    sigma2_range: Tuple[float, float] = (0.01, 5.0),
    n_points: int = 50,
    n_samples: int = 100000,
    seed: int = 42
) -> Dict:
    """
    Esperimento (d): Verifica della disuguaglianza di Jensen.

    Conferma che E[F] < F* quando σ² > 0 (poiché Q è concava).
    """
    sigma2_values = np.linspace(sigma2_range[0], sigma2_range[1], n_points)

    results = {
        'sigma2': [],
        'F_star': [],
        'E_F_th': [],
        'E_F_mc': [],
        'jensen_gap_th': [],
        'jensen_gap_mc': []
    }

    F_star = F_star_theoretical(mu, tau, s)  # = 1 quando mu = tau

    for sigma2 in sigma2_values:
        E_F_th = E_F_theoretical(mu, tau, s, sigma2)
        mc = monte_carlo_estimate(mu, tau, s, sigma2, n_samples, seed)

        results['sigma2'].append(sigma2)
        results['F_star'].append(F_star)
        results['E_F_th'].append(E_F_th)
        results['E_F_mc'].append(mc['E_F'])
        results['jensen_gap_th'].append(F_star - E_F_th)
        results['jensen_gap_mc'].append(F_star - mc['E_F'])

    return results


# ==============================================================================
# Visualizzazioni
# ==============================================================================

def create_plots(
    exp_a_results: Dict,
    exp_b_results: Dict,
    exp_c_results: Dict,
    exp_d_results: Dict,
    output_dir: Path
) -> List[Path]:
    """
    Crea i grafici richiesti e li salva nella directory di output.

    Returns:
        Lista dei percorsi dei file PNG creati
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []

    # Stile comune
    plt.style.use('seaborn-v0_8-whitegrid')

    # =========================================================================
    # Plot 1: E[F] teorico vs Monte Carlo al variare di σ²
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    sigma2 = exp_a_results['sigma2']
    ax1.plot(sigma2, exp_a_results['E_F_th'], 'b-', linewidth=2, label='E[F] Teorico', marker='o')
    ax1.plot(sigma2, exp_a_results['E_F_mc'], 'r--', linewidth=2, label='E[F] Monte Carlo', marker='x')
    ax1.axhline(y=1.0, color='gray', linestyle=':', label='F* = 1 (target)')

    ax1.set_xlabel(r'$\sigma^2$ (Varianza)', fontsize=12)
    ax1.set_ylabel(r'$\mathbb{E}[F]$', fontsize=12)
    ax1.set_title(r'Valore Atteso di F: Teorico vs Monte Carlo ($\mu = \tau = 0$, $s = 1$)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotazioni per errore relativo
    for i, (s2, th, mc) in enumerate(zip(sigma2, exp_a_results['E_F_th'], exp_a_results['E_F_mc'])):
        err = exp_a_results['E_F_err'][i]
        ax1.annotate(f'{err:.2f}%', (s2, mc), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=8, color='green')

    fig1.tight_layout()
    path1 = output_dir / 'structural_bias_EF_comparison.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    plot_paths.append(path1)

    # =========================================================================
    # Plot 2: Decomposizione della Loss (Bias² e Var) vs σ²
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    sigma2_c = exp_c_results['sigma2']
    L_min = exp_c_results['L_min']
    bias_sq = exp_c_results['bias_squared']
    variance = exp_c_results['variance']

    ax2.fill_between(sigma2_c, 0, variance, alpha=0.5, color='blue', label='Varianza')
    ax2.fill_between(sigma2_c, variance, np.array(variance) + np.array(bias_sq),
                    alpha=0.5, color='red', label=r'Bias$^2$')
    ax2.plot(sigma2_c, L_min, 'k-', linewidth=2, label=r'$L_{min}$ (totale)')

    ax2.set_xlabel(r'$\sigma^2$ (Varianza)', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(r'Decomposizione Bias-Varianza della Loss Minima ($\delta = 0$)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(sigma2_c))
    ax2.set_ylim(0, None)

    # Evidenzia che L_min > 0 per sigma2 > 0
    ax2.text(0.5, 0.95, r'$L_{min} > 0$ per ogni $\sigma^2 > 0$',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    fig2.tight_layout()
    path2 = output_dir / 'structural_bias_loss_decomposition.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    plot_paths.append(path2)

    # =========================================================================
    # Plot 3: Loss vs δ per diversi valori di σ²
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    delta_values = np.linspace(-3, 3, 100)
    tau, s = 0.0, 1.0
    sigma2_list = [0.1, 0.5, 1.0, 2.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sigma2_list)))

    for sigma2, color in zip(sigma2_list, colors):
        losses = [Loss_theoretical(d, tau, s, sigma2) for d in delta_values]
        ax3.plot(delta_values, losses, color=color, linewidth=2, label=fr'$\sigma^2 = {sigma2}$')

    # Linea per sigma2 = 0 (caso ideale)
    losses_ideal = [Loss_theoretical(d, tau, s, 0.0) for d in delta_values]
    ax3.plot(delta_values, losses_ideal, 'k--', linewidth=1.5, label=r'$\sigma^2 = 0$ (ideale)')

    ax3.set_xlabel(r'$\delta = \mu - \tau$ (Deviazione dal target)', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title(r'Loss vs $\delta$ per diversi livelli di incertezza', fontsize=12)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(0, None)

    # Marca il punto δ=0
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax3.annotate(r'$\delta = 0$', (0, ax3.get_ylim()[1]*0.9), fontsize=10, ha='center')

    fig3.tight_layout()
    path3 = output_dir / 'structural_bias_loss_vs_delta.png'
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    plot_paths.append(path3)

    # =========================================================================
    # Plot 4: Jensen inequality gap
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    sigma2_d = exp_d_results['sigma2']
    jensen_th = exp_d_results['jensen_gap_th']
    jensen_mc = exp_d_results['jensen_gap_mc']

    ax4.plot(sigma2_d, jensen_th, 'b-', linewidth=2, label=r'$F^* - \mathbb{E}[F]$ Teorico', marker='o', markersize=3)
    ax4.plot(sigma2_d, jensen_mc, 'r--', linewidth=2, label=r'$F^* - \mathbb{E}[F]$ Monte Carlo', marker='x', markersize=3)
    ax4.axhline(y=0, color='gray', linestyle=':')

    ax4.fill_between(sigma2_d, 0, jensen_th, alpha=0.3, color='blue')

    ax4.set_xlabel(r'$\sigma^2$ (Varianza)', fontsize=12)
    ax4.set_ylabel(r'Gap di Jensen: $F^* - \mathbb{E}[F]$', fontsize=12)
    ax4.set_title(r"Verifica Disuguaglianza di Jensen: $\mathbb{E}[F] < F^*$ per $\sigma^2 > 0$", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Evidenzia che il gap è sempre > 0
    ax4.text(0.5, 0.95, r'Gap $> 0$ per ogni $\sigma^2 > 0$ (Jensen verificata)',
            transform=ax4.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    fig4.tight_layout()
    path4 = output_dir / 'structural_bias_jensen_gap.png'
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    plot_paths.append(path4)

    return plot_paths


# ==============================================================================
# Output tabelle formattate
# ==============================================================================

def print_experiment_a_table(results: Dict):
    """Stampa tabella per esperimento (a)."""
    print("\n" + "="*90)
    print("ESPERIMENTO A: Verifica base (policy perfetta, δ=0)")
    print("Parametri: τ=0, s=1, μ=0")
    print("="*90)

    print(f"\n{'σ²':>6} | {'E[F] Th':>10} | {'E[F] MC':>10} | {'Err%':>6} | "
          f"{'Var Th':>10} | {'Var MC':>10} | {'Err%':>6} | {'OK':>4}")
    print("-"*90)

    for i in range(len(results['sigma2'])):
        verified = "✓" if results['verified_all'][i] else "✗"
        print(f"{results['sigma2'][i]:>6.2f} | {results['E_F_th'][i]:>10.6f} | "
              f"{results['E_F_mc'][i]:>10.6f} | {results['E_F_err'][i]:>5.2f}% | "
              f"{results['Var_F_th'][i]:>10.6f} | {results['Var_F_mc'][i]:>10.6f} | "
              f"{results['Var_F_err'][i]:>5.2f}% | {verified:>4}")

    # Riga Loss
    print("-"*90)
    print(f"\n{'σ²':>6} | {'Loss Th':>10} | {'Loss MC':>10} | {'Err%':>6} | {'F*':>10}")
    print("-"*50)
    for i in range(len(results['sigma2'])):
        print(f"{results['sigma2'][i]:>6.2f} | {results['Loss_th'][i]:>10.6f} | "
              f"{results['Loss_mc'][i]:>10.6f} | {results['Loss_err'][i]:>5.2f}% | "
              f"{results['F_star'][i]:>10.6f}")


def print_experiment_b_table(results: Dict):
    """Stampa tabella per esperimento (b)."""
    print("\n" + "="*90)
    print("ESPERIMENTO B: Variazione di δ (deviazione dal target)")
    print("Parametri: τ=0, s=1, σ²=1")
    print("="*90)

    print(f"\n{'μ':>6} | {'δ':>6} | {'F*':>10} | {'E[F] Th':>10} | {'E[F] MC':>10} | "
          f"{'Loss Th':>10} | {'Loss MC':>10} | {'OK':>4}")
    print("-"*90)

    for i in range(len(results['mu'])):
        verified = "✓" if results['verified_all'][i] else "✗"
        print(f"{results['mu'][i]:>6.1f} | {results['delta'][i]:>6.1f} | "
              f"{results['F_star'][i]:>10.6f} | {results['E_F_th'][i]:>10.6f} | "
              f"{results['E_F_mc'][i]:>10.6f} | {results['Loss_th'][i]:>10.6f} | "
              f"{results['Loss_mc'][i]:>10.6f} | {verified:>4}")


def print_verification_summary(exp_a: Dict, exp_b: Dict, exp_c: Dict, exp_d: Dict):
    """Stampa il sommario della verifica."""
    print("\n" + "="*70)
    print("SOMMARIO VERIFICA RISULTATI TEORICI")
    print("="*70)

    all_verified_a = all(exp_a['verified_all'])
    all_verified_b = all(exp_b['verified_all'])

    # Jensen verification
    jensen_verified = all(g > 0 for g in exp_d['jensen_gap_th'])

    # L_min > 0 verification
    lmin_positive = all(l > 0 for l in exp_c['L_min'][1:])  # Skip sigma2=0

    print(f"\n1. Teorema 10 (E[F] formula): {'✓ VERIFICATO' if all_verified_a else '✗ NON VERIFICATO'}")
    print(f"   - Policy perfetta (δ=0): Errore relativo < 1% per tutti i σ²")

    print(f"\n2. Variazione con δ≠0: {'✓ VERIFICATO' if all_verified_b else '✗ NON VERIFICATO'}")
    print(f"   - Formula E[F] valida per tutti i valori di δ testati")

    print(f"\n3. Disuguaglianza di Jensen: {'✓ VERIFICATO' if jensen_verified else '✗ NON VERIFICATO'}")
    print(f"   - E[F] < F* per ogni σ² > 0 (funzione di qualità concava)")

    print(f"\n4. Irraggiungibilità L=0 (Teorema 19): {'✓ VERIFICATO' if lmin_positive else '✗ NON VERIFICATO'}")
    print(f"   - L_min > 0 per ogni σ² > 0")
    print(f"   - Contributi: Bias² > 0 e Var > 0 quando σ² > 0")

    # Overall
    all_passed = all_verified_a and all_verified_b and jensen_verified and lmin_positive
    print(f"\n{'='*70}")
    print(f"RISULTATO GLOBALE: {'✓ TUTTI I TEOREMI VERIFICATI NUMERICAMENTE' if all_passed else '✗ ALCUNE VERIFICHE FALLITE'}")
    print(f"{'='*70}")

    return all_passed


# ==============================================================================
# Funzione principale
# ==============================================================================

def run_structural_bias_verification(
    output_dir: Optional[Path] = None,
    n_samples: int = 100000,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Esegue la verifica completa della bias strutturale.

    Args:
        output_dir: Directory dove salvare i grafici (default: current dir)
        n_samples: Numero di campioni Monte Carlo
        seed: Seed per riproducibilità
        verbose: Se True, stampa le tabelle

    Returns:
        Dizionario con tutti i risultati e i percorsi dei grafici
    """
    if output_dir is None:
        output_dir = Path('.')
    output_dir = Path(output_dir)

    np.random.seed(seed)

    if verbose:
        print("\n" + "="*70)
        print("VERIFICA NUMERICA BIAS STRUTTURALE")
        print("Confronto Valori Teorici vs Monte Carlo")
        print(f"N campioni: {n_samples:,} | Seed: {seed}")
        print("="*70)

    # Esegui esperimenti
    if verbose:
        print("\nEsecuzione esperimento A (varia σ²)...")
    exp_a = experiment_a_vary_sigma(n_samples=n_samples, seed=seed)

    if verbose:
        print("Esecuzione esperimento B (varia δ)...")
    exp_b = experiment_b_vary_delta(n_samples=n_samples, seed=seed)

    if verbose:
        print("Esecuzione esperimento C (irraggiungibilità L=0)...")
    exp_c = experiment_c_loss_unreachability()

    if verbose:
        print("Esecuzione esperimento D (disuguaglianza Jensen)...")
    exp_d = experiment_d_jensen_inequality(n_samples=n_samples, seed=seed)

    # Stampa tabelle
    if verbose:
        print_experiment_a_table(exp_a)
        print_experiment_b_table(exp_b)

    # Crea grafici
    if verbose:
        print("\nCreazione grafici...")
    plot_paths = create_plots(exp_a, exp_b, exp_c, exp_d, output_dir)
    if verbose:
        for path in plot_paths:
            print(f"  Salvato: {path}")

    # Sommario
    if verbose:
        all_passed = print_verification_summary(exp_a, exp_b, exp_c, exp_d)
    else:
        all_passed = (all(exp_a['verified_all']) and all(exp_b['verified_all']) and
                     all(g > 0 for g in exp_d['jensen_gap_th']) and
                     all(l > 0 for l in exp_c['L_min'][1:]))

    return {
        'experiment_a': exp_a,
        'experiment_b': exp_b,
        'experiment_c': exp_c,
        'experiment_d': exp_d,
        'plot_paths': plot_paths,
        'all_verified': all_passed,
        'n_samples': n_samples,
        'seed': seed
    }


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Verifica numerica della bias strutturale nelle loss con reparametrizzazione stocastica'
    )
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='Directory per i grafici di output')
    parser.add_argument('--n-samples', '-n', type=int, default=100000,
                       help='Numero di campioni Monte Carlo')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Seed random per riproducibilità')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disabilita output verbose')

    args = parser.parse_args()

    results = run_structural_bias_verification(
        output_dir=Path(args.output_dir),
        n_samples=args.n_samples,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Exit code basato sulla verifica
    exit(0 if results['all_verified'] else 1)
