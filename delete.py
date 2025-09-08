import zarr
from pathlib import Path

def print_zarr_contents(store, group=None, prefix=""):
    if group is None:
        group = store

    # Print attributes of the current group
    print(f"{prefix}Group attributes: {group.attrs.asdict()}")

    # Print arrays in this group
    for name, array in group.arrays():
        print(f"{prefix}Array: '{name}'")
        print(f"{prefix}  Shape: {array.shape}")
        print(f"{prefix}  Dtype: {array.dtype}")
        print(f"{prefix}  Chunks: {array.chunks}")
        # Print a small sample of values (first 5 elements, flattened)
        flat = array[:].flatten()
        print(f"{prefix}  Sample values: {flat[:5]}")

    # Recursively print subgroups
    for name, subgroup in group.groups():
        print_zarr_contents(store, subgroup, prefix + "  ")

def main():
    zarr_path = Path("198__activ_25089.zarr")
    if not zarr_path.exists():
        print(f"Zarr directory '{zarr_path}' does not exist.")
        return

    store = zarr.open(str(zarr_path), mode="r")
    print_zarr_contents(store)

if __name__ == "__main__":
    main()