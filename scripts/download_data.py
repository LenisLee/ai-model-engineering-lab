import yaml
from datasets import load_dataset


def main():
    with open("configs/base/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ds_name = cfg["data"]["dataset_name"]
    print(f"Downloading {ds_name} ...")
    ds = load_dataset(ds_name)
    ds.save_to_disk(f"data/raw/{ds_name}")
    print(f"Saved to data/raw/{ds_name}")


if __name__ == "__main__":
    main()
