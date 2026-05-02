"""Defense configuration parsing and validation.

Parses the 'defense' section from ConnaCF attack YAML configs and validates
all parameters, raising descriptive errors for invalid values.

Example YAML defense section::

    defense:
        enable_defense: true
        defense_method: "g-safeguard"
        gnn_checkpoint_path: "defense/checkpoints/best_model.pth"
        detection_frequency: 1
        intervention_strategy: "filter"
        gnn_config:
            hidden_dim: 1024
            num_heads: 8
            num_layers: 2
            dropout: 0.2
            embedding_dim: 384
            max_temporal_windows: 3
            aggr_type: "mean"
        blindguard:
            top_k: 3
        g_safeguard:
            threshold: 0.5
        collect_training_data: true
        training_data_output: "defense/training_data/dataset.pkl"
        metrics:
            enabled: true
            plot_detection: true
"""

from dataclasses import dataclass, field


VALID_DEFENSE_METHODS = ("g-safeguard", "blindguard", "t-guard", "m-guard")
VALID_INTERVENTION_STRATEGIES = ("filter", "flag")
VALID_AGGR_TYPES = ("mean", "last")


@dataclass
class GNNConfig:
    """GNN model hyperparameters.

    Attributes:
        hidden_dim: Hidden dimension for GAT layers.
        num_heads: Number of attention heads.
        num_layers: Number of GAT layers.
        dropout: Dropout probability.
        embedding_dim: Dimension of sentence embeddings.
        max_temporal_windows: Number of temporal windows for edge attributes.
        aggr_type: Temporal aggregation strategy ('mean' or 'last').
    """

    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.2
    embedding_dim: int = 384
    max_temporal_windows: int = 3
    aggr_type: str = "mean"


@dataclass
class MetricsConfig:
    """Metrics collection configuration.

    Attributes:
        enabled: Whether to collect defense metrics.
        plot_detection: Whether to generate detection accuracy plots.
    """

    enabled: bool = True
    plot_detection: bool = True


@dataclass
class DefenseConfig:
    """Complete defense configuration parsed from YAML.

    Attributes:
        enable_defense: Whether the defense subsystem is active.
        defense_method: Detection strategy ('g-safeguard' or 'blindguard').
        gnn_checkpoint_path: Path to trained GNN model checkpoint.
        detection_frequency: Run detection every N training rounds.
        intervention_strategy: How to handle detected attackers ('filter' or 'flag').
        gnn_config: GNN model hyperparameters.
        top_k: Number of expected attackers for BlindGuard mode.
        threshold: Classification threshold for G-safeguard mode.
        collect_training_data: Whether to collect training data during runs.
        training_data_output: Path for serialized training data.
        metrics: Metrics collection configuration.
    """

    enable_defense: bool = False
    defense_method: str = "g-safeguard"
    gnn_checkpoint_path: str = ""
    detection_frequency: int = 1
    intervention_strategy: str = "filter"
    gnn_config: GNNConfig = field(default_factory=GNNConfig)
    top_k: int = 3
    threshold: float = 0.5
    symmetric_suppression: bool = False
    collect_training_data: bool = True
    training_data_output: str = "defense/training_data/dataset.pkl"
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    # T-Guard specific
    tguard_decay_factor: float = 0.05
    tguard_quarantine_threshold: float = 0.8
    tguard_restrict_threshold: float = 0.5
    # BlindGuard auto-train
    blindguard_auto_train_turns: int = 3
    blindguard_checkpoint_dir: str = "defense/checkpoints/blindguard"
    # G-Safeguard auto-train
    gsafeguard_auto_train_turns: int = 30
    # M-Guard specific (MASTER defenses)
    mguard_prompt_leakage: bool = True
    mguard_hierarchical: bool = True
    mguard_preemptive: bool = True
    mguard_domain: str = "movie_recommender"


def parse_defense_config(config_dict: dict) -> DefenseConfig:
    """Parse a defense configuration dictionary into a DefenseConfig.

    Extracts values from the dictionary, applying defaults for missing keys.
    After parsing, validates all parameters.

    Args:
        config_dict: Dictionary from the 'defense' section of a YAML config.
            May be empty or None, in which case defaults are used.

    Returns:
        Validated DefenseConfig instance.

    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if not config_dict:
        config_dict = {}

    # Parse GNN config sub-section
    gnn_dict = config_dict.get("gnn_config", {}) or {}
    gnn_config = GNNConfig(
        hidden_dim=gnn_dict.get("hidden_dim", GNNConfig.hidden_dim),
        num_heads=gnn_dict.get("num_heads", GNNConfig.num_heads),
        num_layers=gnn_dict.get("num_layers", GNNConfig.num_layers),
        dropout=gnn_dict.get("dropout", GNNConfig.dropout),
        embedding_dim=gnn_dict.get("embedding_dim", GNNConfig.embedding_dim),
        max_temporal_windows=gnn_dict.get(
            "max_temporal_windows", GNNConfig.max_temporal_windows
        ),
        aggr_type=gnn_dict.get("aggr_type", GNNConfig.aggr_type),
    )

    # Parse metrics sub-section
    metrics_dict = config_dict.get("metrics", {}) or {}
    metrics_config = MetricsConfig(
        enabled=metrics_dict.get("enabled", MetricsConfig.enabled),
        plot_detection=metrics_dict.get("plot_detection", MetricsConfig.plot_detection),
    )

    # Extract BlindGuard top_k from nested or top-level
    blindguard_dict = config_dict.get("blindguard", {}) or {}
    top_k = blindguard_dict.get("top_k", DefenseConfig.top_k)

    # Extract G-safeguard threshold from nested or top-level
    gsafeguard_dict = config_dict.get("g_safeguard", {}) or {}
    threshold = gsafeguard_dict.get("threshold", DefenseConfig.threshold)

    # Parse T-Guard sub-section
    tguard_dict = config_dict.get("t_guard", {}) or {}

    # Parse BlindGuard auto-train sub-section
    blindguard_auto = config_dict.get("blindguard_auto_train", {}) or {}

    # Parse M-Guard sub-section
    mguard_dict = config_dict.get("m_guard", {}) or {}

    config = DefenseConfig(
        enable_defense=config_dict.get("enable_defense", DefenseConfig.enable_defense),
        defense_method=config_dict.get("defense_method", DefenseConfig.defense_method),
        gnn_checkpoint_path=config_dict.get(
            "gnn_checkpoint_path", DefenseConfig.gnn_checkpoint_path
        ),
        detection_frequency=config_dict.get(
            "detection_frequency", DefenseConfig.detection_frequency
        ),
        intervention_strategy=config_dict.get(
            "intervention_strategy", DefenseConfig.intervention_strategy
        ),
        gnn_config=gnn_config,
        top_k=top_k,
        threshold=threshold,
        symmetric_suppression=config_dict.get(
            "symmetric_suppression", DefenseConfig.symmetric_suppression
        ),
        collect_training_data=config_dict.get(
            "collect_training_data", DefenseConfig.collect_training_data
        ),
        training_data_output=config_dict.get(
            "training_data_output", DefenseConfig.training_data_output
        ),
        metrics=metrics_config,
        tguard_decay_factor=tguard_dict.get("decay_factor", DefenseConfig.tguard_decay_factor),
        tguard_quarantine_threshold=tguard_dict.get(
            "taint_threshold_quarantine", DefenseConfig.tguard_quarantine_threshold
        ),
        tguard_restrict_threshold=tguard_dict.get(
            "taint_threshold_restrict", DefenseConfig.tguard_restrict_threshold
        ),
        blindguard_auto_train_turns=blindguard_auto.get(
            "turns", DefenseConfig.blindguard_auto_train_turns
        ),
        blindguard_checkpoint_dir=blindguard_auto.get(
            "checkpoint_dir", DefenseConfig.blindguard_checkpoint_dir
        ),
        gsafeguard_auto_train_turns=config_dict.get(
            "gsafeguard_auto_train_turns", DefenseConfig.gsafeguard_auto_train_turns
        ),
        mguard_prompt_leakage=mguard_dict.get(
            "prompt_leakage", DefenseConfig.mguard_prompt_leakage
        ),
        mguard_hierarchical=mguard_dict.get(
            "hierarchical", DefenseConfig.mguard_hierarchical
        ),
        mguard_preemptive=mguard_dict.get(
            "preemptive", DefenseConfig.mguard_preemptive
        ),
        mguard_domain=mguard_dict.get("domain", DefenseConfig.mguard_domain),
    )

    validate_defense_config(config)
    return config


def validate_defense_config(config: DefenseConfig) -> None:
    """Validate all defense configuration parameters.

    Checks that each parameter falls within its allowed range or set of
    valid values. Raises a descriptive ValueError on the first invalid
    parameter encountered.

    Args:
        config: DefenseConfig instance to validate.

    Raises:
        ValueError: If any parameter is invalid, with a message describing
            the invalid value and the allowed values/range.
    """
    # defense_method
    if config.defense_method not in VALID_DEFENSE_METHODS:
        raise ValueError(
            f"Invalid defense_method '{config.defense_method}'. "
            f"Must be one of {VALID_DEFENSE_METHODS}."
        )

    # intervention_strategy
    if config.intervention_strategy not in VALID_INTERVENTION_STRATEGIES:
        raise ValueError(
            f"Invalid intervention_strategy '{config.intervention_strategy}'. "
            f"Must be one of {VALID_INTERVENTION_STRATEGIES}."
        )

    # detection_frequency must be a positive integer
    if not isinstance(config.detection_frequency, int) or config.detection_frequency < 1:
        raise ValueError(
            f"Invalid detection_frequency '{config.detection_frequency}'. "
            f"Must be a positive integer (>= 1)."
        )

    # top_k must be a positive integer
    if not isinstance(config.top_k, int) or config.top_k < 1:
        raise ValueError(
            f"Invalid top_k '{config.top_k}'. "
            f"Must be a positive integer (>= 1)."
        )

    # threshold must be in [0, 1]
    if not isinstance(config.threshold, (int, float)) or not (0 <= config.threshold <= 1):
        raise ValueError(
            f"Invalid threshold '{config.threshold}'. "
            f"Must be a number in the range [0, 1]."
        )

    # GNN config validation (not needed for t-guard or m-guard)
    if config.defense_method not in ("t-guard", "m-guard"):
        _validate_gnn_config(config.gnn_config)


def _validate_gnn_config(gnn_config: GNNConfig) -> None:
    """Validate GNN-specific configuration parameters.

    Args:
        gnn_config: GNNConfig instance to validate.

    Raises:
        ValueError: If any GNN parameter is invalid.
    """
    if gnn_config.aggr_type not in VALID_AGGR_TYPES:
        raise ValueError(
            f"Invalid gnn_config.aggr_type '{gnn_config.aggr_type}'. "
            f"Must be one of {VALID_AGGR_TYPES}."
        )

    if not isinstance(gnn_config.hidden_dim, int) or gnn_config.hidden_dim < 1:
        raise ValueError(
            f"Invalid gnn_config.hidden_dim '{gnn_config.hidden_dim}'. "
            f"Must be a positive integer."
        )

    if not isinstance(gnn_config.num_heads, int) or gnn_config.num_heads < 1:
        raise ValueError(
            f"Invalid gnn_config.num_heads '{gnn_config.num_heads}'. "
            f"Must be a positive integer."
        )

    if not isinstance(gnn_config.num_layers, int) or gnn_config.num_layers < 1:
        raise ValueError(
            f"Invalid gnn_config.num_layers '{gnn_config.num_layers}'. "
            f"Must be a positive integer."
        )

    if not isinstance(gnn_config.dropout, (int, float)) or not (0 <= gnn_config.dropout <= 1):
        raise ValueError(
            f"Invalid gnn_config.dropout '{gnn_config.dropout}'. "
            f"Must be a number in the range [0, 1]."
        )

    if not isinstance(gnn_config.embedding_dim, int) or gnn_config.embedding_dim < 1:
        raise ValueError(
            f"Invalid gnn_config.embedding_dim '{gnn_config.embedding_dim}'. "
            f"Must be a positive integer."
        )

    if (
        not isinstance(gnn_config.max_temporal_windows, int)
        or gnn_config.max_temporal_windows < 1
    ):
        raise ValueError(
            f"Invalid gnn_config.max_temporal_windows '{gnn_config.max_temporal_windows}'. "
            f"Must be a positive integer."
        )
