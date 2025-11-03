"""
🎯 Template per creare il TUO dataset SCM personalizzato

Modifica questo file per creare un dataset con le TUE relazioni causali!
"""

from scm import SCMDataset, NodeSpec, GroupNoise

# ===========================================================================
# ESEMPIO 1: Dataset semplice con 2 input e 1 output
# ===========================================================================

ds_simple_example = SCMDataset(
    name="simple_2_to_1",
    description="Due input che causano un output",
    tags=["simple", "linear"],

    specs=[
        # Input 1: Temperatura (variabile indipendente)
        NodeSpec("Temperature", [], "eps_Temperature"),

        # Input 2: Umidità (variabile indipendente)
        NodeSpec("Humidity", [], "eps_Humidity"),

        # Output: Consumo energia (dipende da Temperature e Humidity)
        NodeSpec("Energy", ["Temperature", "Humidity"],
                 "3*Temperature + 1.5*Humidity + eps_Energy"),
        #        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #        Consumo aumenta con temperatura e umidità
    ],

    params={},

    # Distribuzione del rumore
    singles={
        "Temperature": lambda rng, n: rng.normal(20, 5, n),  # ~20°C ± 5
        "Humidity": lambda rng, n: rng.normal(60, 15, n),    # ~60% ± 15
        "Energy": lambda rng, n: rng.normal(0, 2, n),        # Rumore piccolo
    },

    groups=None,

    input_labels=["Temperature", "Humidity"],
    target_labels=["Energy"]
)


# ===========================================================================
# ESEMPIO 2: Dataset con relazione non-lineare
# ===========================================================================

ds_nonlinear_example = SCMDataset(
    name="nonlinear_3_to_1",
    description="Relazioni quadratiche e interazioni",
    tags=["nonlinear", "complex"],

    specs=[
        # Inputs esogeni
        NodeSpec("X1", [], "eps_X1"),
        NodeSpec("X2", [], "eps_X2"),
        NodeSpec("X3", [], "eps_X3"),

        # Output con relazione non-lineare
        NodeSpec("Y", ["X1", "X2", "X3"],
                 "X1**2 + 2*X2 + X1*X3 + eps_Y"),
        #        ^^^^^^   ^^^^   ^^^^^
        #        quad.  linear  interaz.
    ],

    params={},

    singles={
        "X1": lambda rng, n: rng.uniform(-2, 2, n),
        "X2": lambda rng, n: rng.standard_normal(n),
        "X3": lambda rng, n: rng.standard_normal(n),
        "Y": lambda rng, n: rng.normal(0, 0.5, n),
    },

    groups=None,

    input_labels=["X1", "X2", "X3"],
    target_labels=["Y"]
)


# ===========================================================================
# ESEMPIO 3: Dataset con parametri configurabili
# ===========================================================================

ds_parametric_example = SCMDataset(
    name="parametric_example",
    description="Coefficienti che puoi cambiare facilmente",
    tags=["parametric"],

    specs=[
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Z", [], "eps_Z"),

        # Usa parametri invece di numeri fissi
        NodeSpec("Y", ["X", "Z"], "w1*X + w2*Z + bias + eps_Y"),
        #                          ^^   ^^   ^^^^
        #                          parametri configurabili
    ],

    # 👉 Cambia questi valori per modificare il comportamento!
    params={
        "w1": 2.5,      # Peso per X
        "w2": 1.8,      # Peso per Z
        "bias": 10.0,   # Offset costante
    },

    singles={
        "X": lambda rng, n: rng.standard_normal(n),
        "Z": lambda rng, n: rng.standard_normal(n),
        "Y": lambda rng, n: rng.normal(0, 0.1, n),
    },

    groups=None,

    input_labels=["X", "Z"],
    target_labels=["Y"]
)


# ===========================================================================
# ESEMPIO 4: Dataset con variabili intermedie (come quello attuale)
# ===========================================================================

ds_hierarchical_example = SCMDataset(
    name="hierarchical_example",
    description="Parents → Children → Output",
    tags=["hierarchical"],

    specs=[
        # Layer 1: Parents (exogenous)
        NodeSpec("P1", [], "eps_P1"),
        NodeSpec("P2", [], "eps_P2"),
        NodeSpec("P3", [], "eps_P3"),

        # Layer 2: Children (dipendono dai parents)
        NodeSpec("C1", ["P1"], "2*P1 + eps_C1"),
        NodeSpec("C2", ["P2"], "1.5*P2 + eps_C2"),
        NodeSpec("C3", ["P3"], "3*P3 + eps_C3"),

        # Layer 3: Output finale
        NodeSpec("Y", ["C1", "C2", "C3"], "C1 + C2 + C3 + eps_Y"),
    ],

    params={},

    singles={
        "P1": lambda rng, n: rng.standard_normal(n),
        "P2": lambda rng, n: rng.standard_normal(n),
        "P3": lambda rng, n: rng.standard_normal(n),
        "C1": lambda rng, n: rng.normal(0, 0.5, n),
        "C2": lambda rng, n: rng.normal(0, 0.5, n),
        "C3": lambda rng, n: rng.normal(0, 0.5, n),
        "Y": lambda rng, n: rng.normal(0, 1, n),
    },

    groups=None,

    input_labels=["P1", "P2", "P3", "C1", "C2", "C3"],
    target_labels=["Y"]
)


# ===========================================================================
# ESEMPIO 5: Dataset con rumore correlato
# ===========================================================================

ds_correlated_example = SCMDataset(
    name="correlated_noise",
    description="Variabili con rumore correlato (confounding)",
    tags=["correlated", "confounding"],

    specs=[
        # X e Z hanno rumore correlato (confounding latente)
        NodeSpec("X", [], "eps_X"),
        NodeSpec("Z", [], "eps_Z"),

        # Y dipende da entrambi
        NodeSpec("Y", ["X", "Z"], "X + Z + eps_Y"),
    ],

    params={},

    singles={
        "Y": lambda rng, n: rng.normal(0, 0.1, n),
    },

    # Rumore correlato per X e Z
    groups=[
        GroupNoise(
            nodes=("X", "Z"),
            sampler=lambda rng, n: rng.multivariate_normal(
                mean=[0, 0],
                cov=[[1.0, 0.8],    # Correlazione 0.8 tra X e Z
                     [0.8, 1.0]],
                size=n
            )
        )
    ],

    input_labels=["X", "Z"],
    target_labels=["Y"]
)


# ===========================================================================
# 🎯 TEST: Prova il tuo dataset
# ===========================================================================

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # Scegli quale dataset testare
    dataset = ds_simple_example  # ← Cambia questo!

    print(f"\n{'='*70}")
    print(f"Testing dataset: {dataset.scm.meta['name']}")
    print(f"{'='*70}")

    # Genera campioni
    df = dataset.sample(n=1000, seed=42)

    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"\n📈 Statistics:")
    print(df.describe())

    # Correlazioni
    print(f"\n🔗 Correlations:")
    print(df.corr().round(3))

    # Plot
    if len(dataset.input_labels) == 2 and len(dataset.target_labels) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Input 1 vs Output
        axes[0].scatter(df[dataset.input_labels[0]],
                       df[dataset.target_labels[0]],
                       alpha=0.5)
        axes[0].set_xlabel(dataset.input_labels[0])
        axes[0].set_ylabel(dataset.target_labels[0])
        axes[0].set_title(f"{dataset.input_labels[0]} vs {dataset.target_labels[0]}")

        # Plot 2: Input 2 vs Output
        axes[1].scatter(df[dataset.input_labels[1]],
                       df[dataset.target_labels[0]],
                       alpha=0.5)
        axes[1].set_xlabel(dataset.input_labels[1])
        axes[1].set_ylabel(dataset.target_labels[0])
        axes[1].set_title(f"{dataset.input_labels[1]} vs {dataset.target_labels[0]}")

        plt.tight_layout()
        plt.savefig('/tmp/scm_test.png', dpi=150)
        print(f"\n💾 Plot saved to: /tmp/scm_test.png")

    print(f"\n{'='*70}")
    print(f"✅ Test completed!")
    print(f"{'='*70}\n")
