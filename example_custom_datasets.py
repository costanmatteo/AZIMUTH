"""
Esempi di dataset SCM personalizzati
Copia questo codice in uncertainty_predictor/scm_ds/datasets.py
"""

from scm_ds.scm import *

# ============================================================================
# ESEMPIO 1: Dataset semplice 3 input → 1 output
# ============================================================================

ds_simple_3to1 = SCMDataset(
    name="simple_3to1",
    description="Tre input indipendenti che causano un output",
    tags=None,
    specs=[
        # 3 input indipendenti
        NodeSpec("Temperatura", [], "eps_Temperatura"),
        NodeSpec("Umidita", [], "eps_Umidita"),
        NodeSpec("Pressione", [], "eps_Pressione"),

        # Output che dipende da tutti e 3
        NodeSpec("Consumo", ["Temperatura", "Umidita", "Pressione"],
                 "3*Temperatura + 1.5*Umidita + 0.8*Pressione + eps_Consumo"),
    ],
    params={},
    singles={
        "Temperatura": lambda rng,n: rng.normal(20, 5, n),    # ~20°C ± 5
        "Umidita": lambda rng,n: rng.normal(60, 15, n),       # ~60% ± 15
        "Pressione": lambda rng,n: rng.normal(1013, 10, n),   # ~1013 hPa ± 10
        "Consumo": lambda rng,n: rng.normal(0, 2, n),         # Poco rumore
    },
    groups=None,
    input_labels=["Temperatura", "Umidita", "Pressione"],
    target_labels=["Consumo"]
)


# ============================================================================
# ESEMPIO 2: Dataset con relazione non-lineare (quadratica)
# ============================================================================

ds_nonlinear = SCMDataset(
    name="nonlinear",
    description="Relazione quadratica tra input e output",
    tags=None,
    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Y", ["X"], "X**2 + 2*X + 5 + eps_Y"),  # Y = X² + 2X + 5
    ],
    params={},
    singles={
        "X": lambda rng,n: rng.uniform(-3, 3, n),  # X tra -3 e 3
        "Y": lambda rng,n: rng.normal(0, 0.5, n),  # Poco rumore
    },
    groups=None,
    input_labels=["X"],
    target_labels=["Y"]
)


# ============================================================================
# ESEMPIO 3: Dataset gerarchico (2 layer)
# ============================================================================

ds_hierarchical = SCMDataset(
    name="hierarchical",
    description="Input → Features intermedie → Output",
    tags=None,
    specs=[
        # Layer 1: Input base
        NodeSpec("I1", [], "eps_I1"),
        NodeSpec("I2", [], "eps_I2"),
        NodeSpec("I3", [], "eps_I3"),

        # Layer 2: Features intermedie
        NodeSpec("F1", ["I1", "I2"], "2*I1 + I2 + eps_F1"),
        NodeSpec("F2", ["I2", "I3"], "I2 + 3*I3 + eps_F2"),

        # Layer 3: Output finale
        NodeSpec("Y", ["F1", "F2"], "F1 + F2 + eps_Y"),
    ],
    params={},
    singles={
        "I1": lambda rng,n: rng.standard_normal(n),
        "I2": lambda rng,n: rng.standard_normal(n),
        "I3": lambda rng,n: rng.standard_normal(n),
        "F1": lambda rng,n: rng.normal(0, 0.5, n),
        "F2": lambda rng,n: rng.normal(0, 0.5, n),
        "Y": lambda rng,n: rng.normal(0, 1, n),
    },
    groups=None,
    input_labels=["I1", "I2", "I3", "F1", "F2"],
    target_labels=["Y"]
)


# ============================================================================
# ESEMPIO 4: Dataset con pesi configurabili
# ============================================================================

ds_parametric = SCMDataset(
    name="parametric",
    description="Coefficienti facilmente modificabili",
    tags=None,
    specs=[
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        # Usa parametri invece di numeri fissi
        NodeSpec("Y", ["X1", "X2"], "w1*X1 + w2*X2 + bias + eps_Y"),
    ],
    params={
        "w1": 3.5,    # ← Cambia questo per modificare peso di X1
        "w2": 1.2,    # ← Cambia questo per modificare peso di X2
        "bias": 10.0, # ← Cambia questo per offset
    },
    singles={
        "X1": lambda rng,n: rng.standard_normal(n),
        "X2": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.normal(0, 0.1, n),
    },
    groups=None,
    input_labels=["X1", "X2"],
    target_labels=["Y"]
)


# ============================================================================
# ESEMPIO 5: Dataset simile all'attuale ma semplificato (3 parent invece di 5)
# ============================================================================

ds_simplified = SCMDataset(
    name="simplified_3parents",
    description="Versione semplificata con solo 3 parent",
    tags=None,
    specs=[
        # Parents
        NodeSpec("P1", [], "eps_P1"),
        NodeSpec("P2", [], "eps_P2"),
        NodeSpec("P3", [], "eps_P3"),

        # Children
        NodeSpec("C1", ["P1"], "2*P1 + eps_C1"),
        NodeSpec("C2", ["P2"], "1.5*P2 + eps_C2"),
        NodeSpec("C3", ["P3"], "3*P3 + eps_C3"),

        # Output
        NodeSpec("Y", ["C1", "C2", "C3"], "C1 + C2 + C3 + eps_Y"),
    ],
    params={},
    singles={
        "P1": lambda rng,n: rng.standard_normal(n),
        "P2": lambda rng,n: rng.standard_normal(n),
        "P3": lambda rng,n: rng.standard_normal(n),
        "C1": lambda rng,n: rng.normal(0, 0.5, n),
        "C2": lambda rng,n: rng.normal(0, 0.5, n),
        "C3": lambda rng,n: rng.normal(0, 0.5, n),
        "Y": lambda rng,n: rng.normal(0, 1, n),
    },
    groups=None,
    input_labels=["P1", "P2", "P3", "C1", "C2", "C3"],
    target_labels=["Y"]
)
