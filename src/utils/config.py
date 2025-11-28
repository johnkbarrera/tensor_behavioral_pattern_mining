import yaml
from pathlib import Path


def replace_label(data_dict, val_to_replace, new_val):
    """Recursively replaces '{val_to_replace}' placeholders."""
    old_tag = val_to_replace
    new_tag = new_val
    if isinstance(data_dict, str):
        return data_dict.replace(f"{{{old_tag}}}", new_tag)
    elif isinstance(data_dict, dict):
        return {k: replace_label(v, old_tag, new_tag) for k, v in data_dict.items()}
    elif isinstance(data_dict, list):
        return [replace_label(item, old_tag, new_tag) for item in data_dict]
    return data_dict


def load_config(base_dir):
    """Carga todo el archivo YAML y lo retorna como dict."""
    path = f"{base_dir}/config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    dataset_active = config_data["dataset"]["active"]
    file_name = config_data["dataset"]["filename"]
    config_data = replace_label(config_data, "dataset_active", dataset_active)

    # New Paths
    config_data["paths"][
        "session_records"
    ] = f"{config_data["paths"]["raw"]}{file_name}"

    # Footprints
    config_data["paths"][
        "footprints"
    ] = f"{config_data["paths"]["processed"]}footprints.csv"
    config_data["paths"][
        "tensors_convolution"
    ] = f"{config_data["paths"]["processed"]}tensors_convolution.csv"

    # Mathematical embedding
    config_data["paths"][
        "tensors_svd"
    ] = f"{config_data["paths"]["processed"]}tensors_svd"
    config_data["paths"][
        "tensors_pca"
    ] = f"{config_data["paths"]["processed"]}tensors_pca"

    # Normalization
    config_data["paths"][
        "tensors_normalized_01"
    ] = f"{config_data["paths"]["processed"]}tensors_normalized_01"
    config_data["paths"][
        "tensors_normalized_L1"
    ] = f"{config_data["paths"]["processed"]}tensors_normalized_L1"
    config_data["paths"][
        "tensors_normalized_L2"
    ] = f"{config_data["paths"]["processed"]}tensors_normalized_L2"

    # Cluster
    config_data["paths"][
        "cluster_labels"
    ] = f"{config_data["paths"]["processed"]}cluster_labels"
    config_data["paths"][
        "cluster_centroids"
    ] = f"{config_data["paths"]["processed"]}cluster_centroids.csv"
    config_data["paths"][
        "cluster_representations"
    ] = f"{config_data["paths"]["processed"]}cluster_representations.csv"

    for key, value in config_data["paths"].items():
        config_data["paths"][key] = f"{base_dir}/{value}"

    return config_data
