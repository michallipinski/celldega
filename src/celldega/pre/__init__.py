"""
Module for pre-processing to generate LandscapeFiles from ST data.
"""

try:
    import pyvips
except ImportError:
    pyvips = None

import base64
import colorsys
import hashlib
import json
from pathlib import Path
import subprocess
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from shapely.geometry import MultiPolygon, Point, Polygon
from skimage.io import imread, imsave
import tifffile
import zarr

from .boundary_tile import (
    _round_nested_coord_list,
    make_cell_boundary_tiles,
    make_cell_boundary_tiles_row_groups,
)
from .cell_clusters import (
    _load_xenium_cluster_data,
    _save_cluster_data,
    add_clustering_from_adata,
    cluster_gene_expression,
    create_cluster_and_meta_cluster,
)
from .colors import _create_cluster_colors, _hsv_to_hex
from .geometry import _to_coords, _to_geometry
from .image_info import get_image_info, resolve_xenium_morphology_ome_path
from .image_tiles import (
    _convert_to_jpeg,
    _convert_to_png,
    _convert_to_webp,
    _process_image_channel,
    _reduce_image_size,
    create_image_tiles,
    create_image_tiles_h_and_e,
    create_image_tiles_merscope,
    create_image_tiles_xenium,
    make_deepzoom_pyramid,
    remove_intermediate_files,
)
from .landscape import (
    calc_meta_gene_data,
    read_cbg_mtx,
    save_cbg_gene_parquets,
    save_cbg_gene_parquets_row_groups,
)
from .meta_cell import (
    _convert_long_id_to_short,
    _load_meta_cell_by_technology,
    make_meta_cell_image_coord,
    make_meta_gene,
)
from .nbhd_cloud import (
    write_cell_clusters_meta,
    write_gene_cell_scatter,
    write_gene_shapes,
    write_gene_shapes_streaming,
    write_meta_gene_for_nbhd_cloud,
    write_meta_slice,
    write_nbhd_cloud_cells,
    write_nbhd_cloud_dataset,
    write_nbhd_cloud_shapes_and_features,
)
from .raw_bundle import (
    _check_required_files,
    _xenium_unzipper,
    write_identity_transform,
    write_xenium_transform,
)
from .sbg_tile import write_pseudotranscripts_from_sbg
from .trx_tile import make_trx_tiles, make_trx_tiles_row_groups


def main(*args, **kwargs):
    from .run_pre_processing import main as _main

    return _main(*args, **kwargs)


def pack_image_tiles_to_parquet(
    pyramid_dir,
    channel_name,
    output_path,
    image_format=".webp",
    delete_source_tiles=True,
    max_row_groups_per_file=2000,
):
    """
    Pack all image tiles from a DeepZoom pyramid into chunked parquet files with row groups.

    Each zoom level's tiles are stored as row groups, allowing efficient range-based access.
    The formula for row group index is:
        row_group_index = sum of tiles in previous zoom levels + tile_x * num_tiles_y + tile_y

    For large datasets, tiles are split across multiple parquet files, each containing
    at most `max_row_groups_per_file` row groups.

    Args:
        pyramid_dir (str): Path to the pyramid_images directory.
        channel_name (str): Name of the image channel (e.g., "dapi").
        output_path (str): Path to the output directory (will contain chunk_X.parquet files).
        image_format (str): Image file extension (default ".webp").
        delete_source_tiles (bool): If True, delete the original tile files after packing.
        max_row_groups_per_file (int): Maximum row groups per file (default 400).

    Returns:
        dict: Image tile metadata including grid info per zoom level and image dimensions.
    """
    import xml.etree.ElementTree as ET

    import pyarrow as pa
    import pyarrow.parquet as pq

    tiles_dir = Path(pyramid_dir) / f"{channel_name}_files"

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    # Read image dimensions from .dzi file before potentially deleting it
    dzi_file = Path(pyramid_dir) / f"{channel_name}.dzi"
    image_width = None
    image_height = None
    tile_size = 512  # Default tile size

    if dzi_file.exists():
        try:
            # Read raw content to check format
            dzi_content = dzi_file.read_text()
            print(f"DZI file content preview: {dzi_content[:200]}")

            tree = ET.parse(dzi_file)
            root = tree.getroot()

            # Try different ways to find the Size element
            # Method 1: With namespace
            ns = {"dzi": "http://schemas.microsoft.com/deepzoom/2008"}
            size_elem = root.find(".//dzi:Size", ns)

            # Method 2: Without namespace (pyvips may not include namespace)
            if size_elem is None:
                size_elem = root.find(".//Size")

            # Method 3: Direct child of root
            if size_elem is None:
                size_elem = root.find("Size")

            # Method 4: root might be Image element itself
            if size_elem is None:
                for child in root:
                    if "Size" in child.tag:
                        size_elem = child
                        break

            if size_elem is not None:
                image_width = int(size_elem.get("Width"))
                image_height = int(size_elem.get("Height"))
            else:
                print(
                    f"Warning: Could not find Size element in DZI. Root tag: {root.tag}, children: {[c.tag for c in root]}"
                )

            # Get tile size from Image element
            if root.get("TileSize"):
                tile_size = int(root.get("TileSize"))

            print(
                f"Read image dimensions from DZI: {image_width}x{image_height}, tile_size={tile_size}"
            )
        except Exception as e:
            print(f"Warning: Could not parse DZI file: {e}")
            import traceback

            traceback.print_exc()

    # Discover zoom levels and tiles
    zoom_levels = sorted([int(d.name) for d in tiles_dir.iterdir() if d.is_dir()])

    if not zoom_levels:
        raise ValueError(f"No zoom levels found in {tiles_dir}")

    print(f"Found {len(zoom_levels)} zoom levels: {zoom_levels[0]} to {zoom_levels[-1]}")

    # Collect all tiles with their metadata
    all_tiles = []
    zoom_info = {}

    for zoom in zoom_levels:
        zoom_dir = tiles_dir / str(zoom)
        tile_files = list(zoom_dir.glob(f"*{image_format}"))

        # Parse tile coordinates from filenames (format: x_y.webp)
        tiles_in_zoom = []
        for tile_file in tile_files:
            parts = tile_file.stem.split("_")
            if len(parts) == 2:
                tile_x, tile_y = int(parts[0]), int(parts[1])
                image_bytes = tile_file.read_bytes()
                tiles_in_zoom.append((tile_x, tile_y, image_bytes))

        # Sort tiles in column-major order for consistent indexing
        tiles_in_zoom.sort(key=lambda t: (t[0], t[1]))

        # Calculate grid dimensions
        if tiles_in_zoom:
            max_x = max(t[0] for t in tiles_in_zoom)
            max_y = max(t[1] for t in tiles_in_zoom)
            num_tiles_x = max_x + 1
            num_tiles_y = max_y + 1
        else:
            num_tiles_x = 0
            num_tiles_y = 0

        zoom_info[zoom] = {
            "num_tiles_x": num_tiles_x,
            "num_tiles_y": num_tiles_y,
            "num_tiles": len(tiles_in_zoom),
            "row_group_offset": len(all_tiles),
        }

        all_tiles.extend([(zoom, tx, ty, data) for tx, ty, data in tiles_in_zoom])

    if not all_tiles:
        raise ValueError("No tiles found to pack")

    total_tiles = len(all_tiles)
    num_files = (total_tiles + max_row_groups_per_file - 1) // max_row_groups_per_file
    print(
        f"Packing {total_tiles} tiles into {num_files} parquet files "
        f"(max {max_row_groups_per_file} per file)..."
    )

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create schema
    schema = pa.schema(
        [
            pa.field("zoom", pa.int32()),
            pa.field("tile_x", pa.int32()),
            pa.field("tile_y", pa.int32()),
            pa.field("image_data", pa.binary()),
        ]
    )

    # Add metadata
    metadata = {
        b"storage_mode": b"row_groups_image_chunked",
        b"zoom_info": json.dumps(zoom_info).encode("utf-8"),
        b"channel_name": channel_name.encode("utf-8"),
        b"image_format": image_format.encode("utf-8"),
    }
    schema = schema.with_metadata(metadata)

    # Write tiles to chunked parquet files
    file_list = []
    current_file_idx = 0
    current_row_in_file = 0
    writer = None

    for _tile_idx, (zoom, tile_x, tile_y, image_bytes) in enumerate(all_tiles):
        # Start a new file if needed
        if current_row_in_file == 0 or current_row_in_file >= max_row_groups_per_file:
            if writer is not None:
                writer.close()
            file_name = f"chunk_{current_file_idx}.parquet"
            file_path = output_dir / file_name
            file_list.append(file_name)
            writer = pq.ParquetWriter(str(file_path), schema, write_statistics=False)
            current_file_idx += 1
            current_row_in_file = 0

        # Write this tile as a row group
        tile_table = pa.table(
            {
                "zoom": [zoom],
                "tile_x": [tile_x],
                "tile_y": [tile_y],
                "image_data": [image_bytes],
            },
            schema=schema,
        )
        writer.write_table(tile_table)
        current_row_in_file += 1

    if writer is not None:
        writer.close()

    print(f"Wrote {total_tiles} image tiles across {len(file_list)} files to {output_dir}")

    # Delete source tile images if requested (but keep .dzi files for dimension info)
    if delete_source_tiles:
        import shutil

        print(f"Deleting source tile images from {tiles_dir}...")
        try:
            shutil.rmtree(tiles_dir)
            # Keep the .dzi file - it's tiny and provides image dimensions
            print(f"Deleted source tile images for {channel_name} (kept .dzi file)")
        except Exception as e:
            print(f"Warning: Could not delete source tiles: {e}")

    return {
        "channel_name": channel_name,
        "num_tiles": total_tiles,
        "zoom_levels": zoom_levels,
        "zoom_info": zoom_info,
        "image_width": image_width,
        "image_height": image_height,
        "tile_size": tile_size,
        # Chunk info for frontend
        "directory": channel_name,
        "files": file_list,
        "max_row_groups_per_file": max_row_groups_per_file,
        "total_row_groups": total_tiles,
    }


def make_chromium_from_anndata(adata, path_dega_files):
    """Generate minimal LandscapeFiles from a Chromium AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing scRNA-seq count data.
    path_dega_files : str or Path
        Directory where LandscapeFiles will be written.

    Raises
    ------
    ValueError
        If the expression matrix contains non-integer values.
    """

    print("\n========Process Chromium AnnData========")
    path_dega_files = Path(path_dega_files)
    path_dega_files.mkdir(parents=True, exist_ok=True)

    X = adata.layers.get("counts", adata.X)

    data = X.data if issparse(X) else np.asarray(X)

    if not np.all(np.equal(np.mod(data, 1), 0)):
        raise ValueError("Chromium processing requires integer counts")

    if issparse(X):
        cbg = pd.DataFrame.sparse.from_spmatrix(X, index=adata.obs_names, columns=adata.var_names)
    else:
        cbg = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    cell_meta = pd.DataFrame({"name": adata.obs_names, "geometry": [[0.0, 0.0]] * adata.n_obs})
    cell_meta.to_parquet(path_dega_files / "cell_metadata.parquet", index=False)

    save_cbg_gene_parquets("Chromium", path_dega_files, cbg)

    make_meta_gene(cbg, path_dega_files / "meta_gene.parquet")

    (path_dega_files / "pyramid_images").mkdir(exist_ok=True)

    save_landscape_parameters(
        technology="Chromium",
        path_dega_files=path_dega_files,
        image_name="",
        tile_size=1,
        image_info=[],
    )


def get_max_zoom_level(path_image_pyramid):
    """Returns the maximum zoom level based on the highest-numbered directory in the specified path.

    Args:
        path_image_pyramid (str): Path to the directory containing zoom level directories.

    Returns:
        int: The maximum zoom level.
    """
    path_pyramid = Path(path_image_pyramid)
    zoom_levels = [
        entry.name for entry in path_pyramid.iterdir() if entry.is_dir() and entry.name.isdigit()
    ]
    return max(map(int, zoom_levels)) if zoom_levels else None


def save_landscape_parameters(
    technology,
    path_dega_files,
    image_name="dapi_files",
    tile_size=1000,
    image_info=None,
    image_format=".webp",
    use_int_index=True,
    segmentation_approach="default",
    use_row_groups=False,
    tile_grid_info=None,
    image_tile_info=None,
    trx_chunk_info=None,
    cell_chunk_info=None,
    cbg_chunk_info=None,
):
    """Saves the landscape parameters to a JSON file.

    Args:
        technology (str): The technology used to generate the data.
        path_dega_files (str): Path to the directory where landscape files are stored.
        image_name (str, optional): Name of the image directory. Defaults to "dapi_files".
        tile_size (int, optional): Tile size for the image pyramid. Defaults to 1000.
        image_info (dict, optional): Additional image metadata. Defaults to None.
        image_format (str, optional): Format of the image files. Defaults to ".webp".
        use_int_index (bool, optional): Use integer name for cell_tile and trx_tile.
        use_row_groups (bool, optional): If True, tiles are stored as row groups. Defaults to False.
        tile_grid_info (dict, optional): Tile grid metadata when using row groups.
        image_tile_info (dict, optional): Image tile metadata from pack_image_tiles_to_parquet.
        trx_chunk_info (dict, optional): Chunk info for transcript parquet files.
        cell_chunk_info (dict, optional): Chunk info for cell segmentation parquet files.
        cbg_chunk_info (dict, optional): Chunk info for CBG parquet files.

    Returns:
        None
    """
    print("\n========Save landscape parameters========")

    if image_info is None:
        image_info = {}

    if technology == "h&e":
        image_name = "h_and_e_files"
        image_info = [{"name": "h&e", "button_name": "H&E", "color": [0, 0, 255]}]

    # Get max pyramid zoom - use from image_tile_info if available (row groups mode)
    # since the tile files may have been deleted
    if image_tile_info and use_row_groups:
        # Get max zoom from the first channel's info
        first_channel_info = next(iter(image_tile_info.values()), None)
        if first_channel_info and "zoom_levels" in first_channel_info:
            max_pyramid_zoom = max(first_channel_info["zoom_levels"])
        else:
            max_pyramid_zoom = None
    else:
        path_image_pyramid = Path(path_dega_files) / "pyramid_images" / image_name
        max_pyramid_zoom = get_max_zoom_level(path_image_pyramid)

    path_landscape_parameters = Path(path_dega_files) / "landscape_parameters.json"

    # if technology is 'h&e' set parameters
    if technology == "h&e":
        landscape_parameters = {
            "technology": technology,
            "segmentation_approach": ["N.A."],
            "max_pyramid_zoom": max_pyramid_zoom,
            "tile_size": "N.A.",
            "image_info": image_info,
            "image_format": image_format,
            "use_int_index": "N.A.",
        }
    elif technology != "custom":
        landscape_parameters = {
            "technology": technology,
            "segmentation_approach": [segmentation_approach],
            "max_pyramid_zoom": max_pyramid_zoom,
            "tile_size": tile_size,
            "image_info": image_info,
            "image_format": image_format,
            "use_int_index": use_int_index,
            "use_row_groups": use_row_groups,
        }

        # Add row group specific metadata
        if use_row_groups and tile_grid_info:
            landscape_parameters["row_group_files"] = {}

            # Add chunked CBG files
            if cbg_chunk_info:
                landscape_parameters["row_group_files"]["cbg"] = {
                    "directory": cbg_chunk_info.get("directory", "cbg"),
                    "files": cbg_chunk_info.get("files", []),
                    "max_row_groups_per_file": cbg_chunk_info.get("max_row_groups_per_file", 2000),
                    "total_row_groups": cbg_chunk_info.get("total_row_groups", 0),
                    "gene_to_row_group": cbg_chunk_info.get("gene_to_row_group", {}),
                }
            else:
                # Legacy single file mode (backwards compatibility)
                landscape_parameters["row_group_files"]["cbg"] = "cbg.parquet"

            # Add chunked transcript files
            if trx_chunk_info:
                landscape_parameters["row_group_files"]["transcripts"] = {
                    "directory": "transcripts",
                    "files": trx_chunk_info.get("files", []),
                    "max_row_groups_per_file": trx_chunk_info.get("max_row_groups_per_file", 10000),
                    "total_row_groups": trx_chunk_info.get("total_row_groups", 0),
                }
            else:
                # Legacy single file mode (backwards compatibility)
                landscape_parameters["row_group_files"]["transcripts"] = "transcripts.parquet"

            # Add chunked cell segmentation files
            if cell_chunk_info:
                landscape_parameters["row_group_files"]["cell_segmentation"] = {
                    "directory": "cell_segmentation",
                    "files": cell_chunk_info.get("files", []),
                    "max_row_groups_per_file": cell_chunk_info.get(
                        "max_row_groups_per_file", 10000
                    ),
                    "total_row_groups": cell_chunk_info.get("total_row_groups", 0),
                }
            else:
                # Legacy single file mode (backwards compatibility)
                landscape_parameters["row_group_files"]["cell_segmentation"] = (
                    "cell_segmentation.parquet"
                )

            # Add image parquet files for each channel with zoom info
            # Parquet files are now in pyramid_images/{channel_name}/ directories (chunked)
            pyramid_images_dir = Path(path_dega_files) / "pyramid_images"
            if pyramid_images_dir.exists():
                image_parquets = {}

                # Check for chunked directories (new format)
                for channel_dir in pyramid_images_dir.iterdir():
                    if channel_dir.is_dir():
                        # Sort numerically, not alphabetically (chunk_10 should come after chunk_9)
                        chunk_files = sorted(
                            channel_dir.glob("chunk_*.parquet"),
                            key=lambda f: int(f.stem.split("_")[1]),
                        )
                        if chunk_files:
                            channel_name = channel_dir.name
                            image_entry = {
                                "directory": f"pyramid_images/{channel_name}",
                                "files": [f.name for f in chunk_files],
                            }
                            # Add chunk info and zoom_info if available from image_tile_info
                            if image_tile_info and channel_name in image_tile_info:
                                channel_info = image_tile_info[channel_name]
                                image_entry["zoom_info"] = channel_info.get("zoom_info", {})
                                image_entry["zoom_levels"] = channel_info.get("zoom_levels", [])
                                image_entry["max_row_groups_per_file"] = channel_info.get(
                                    "max_row_groups_per_file", 2000
                                )
                                image_entry["total_row_groups"] = channel_info.get(
                                    "total_row_groups", 0
                                )
                            image_parquets[channel_name] = image_entry

                # Also check for legacy single parquet files (backwards compatibility)
                for pq_file in pyramid_images_dir.glob("*.parquet"):
                    channel_name = pq_file.stem
                    if channel_name not in image_parquets:
                        image_entry = {
                            "path": f"pyramid_images/{pq_file.name}",
                        }
                        # Add zoom_info if available from image_tile_info
                        if image_tile_info and channel_name in image_tile_info:
                            channel_info = image_tile_info[channel_name]
                            image_entry["zoom_info"] = channel_info.get("zoom_info", {})
                            image_entry["zoom_levels"] = channel_info.get("zoom_levels", [])
                        image_parquets[channel_name] = image_entry

                if image_parquets:
                    landscape_parameters["row_group_files"]["images"] = image_parquets

            # Store image dimensions from first channel (all channels have same dimensions)
            if image_tile_info:
                first_channel_info = next(iter(image_tile_info.values()), None)
                if first_channel_info:
                    landscape_parameters["image_dimensions"] = {
                        "width": first_channel_info.get("image_width"),
                        "height": first_channel_info.get("image_height"),
                        "tile_size": first_channel_info.get("tile_size", 512),
                    }

            # Store grid dimensions - frontend computes row group index using:
            # row_group_index = tile_x * num_tiles_y + tile_y
            landscape_parameters["tile_grid"] = {
                "tile_size": tile_grid_info.get("tile_size", tile_size),
                "num_tiles_x": tile_grid_info.get("num_tiles_x"),
                "num_tiles_y": tile_grid_info.get("num_tiles_y"),
                "x_min": tile_grid_info.get("x_min"),
                "x_max": tile_grid_info.get("x_max"),
                "y_min": tile_grid_info.get("y_min"),
                "y_max": tile_grid_info.get("y_max"),
            }
    else:
        with path_landscape_parameters.open() as file:
            landscape_parameters = json.load(file)
        landscape_parameters["segmentation_approach"].append(segmentation_approach)

    with path_landscape_parameters.open("w") as file:
        json.dump(landscape_parameters, file, indent=4)

    print("Done.")


def add_custom_segmentation(
    technology, path_dega_files, path_segmentation_files, image_scale=1, tile_size=250
):
    """
    Add custom segmentation to existing landscape files.

    Parameters:
    - technology: Technology type (e.g., "Xenium", "MERSCOPE", "custom")
    - path_dega_files: Path to landscape files
    - path_segmentation_files: Path to segmentation files
    - image_scale: Image scale factor
    - tile_size: Tile size for processing
    """
    with (Path(path_segmentation_files) / "segmentation_parameters.json").open() as file:
        segmentation_parameters = json.load(file)

    cbg_custom = pd.read_parquet(Path(path_segmentation_files) / "cell_by_gene_matrix.parquet")

    # make sure all genes are present in cbg_custom
    meta_gene = pd.read_parquet(Path(path_dega_files) / "meta_gene.parquet")
    missing_cols = meta_gene.index.difference(cbg_custom.columns)
    for col in missing_cols:
        cbg_custom[col] = 0

    make_meta_gene(
        cbg=cbg_custom,
        path_output=Path(path_dega_files)
        / f"meta_gene_{segmentation_parameters['segmentation_approach']}.parquet",
    )

    make_meta_cell_image_coord(
        technology=segmentation_parameters["technology"],
        path_transformation_matrix=str(Path(path_dega_files) / "micron_to_image_transform.csv"),
        path_meta_cell_micron=str(
            Path(path_segmentation_files) / "cell_metadata_micron_space.parquet"
        ),
        path_meta_cell_image=str(
            Path(path_dega_files)
            / f"cell_metadata_{segmentation_parameters['segmentation_approach']}.parquet"
        ),
        image_scale=image_scale,
    )

    save_cbg_gene_parquets(
        technology=technology,
        base_path=path_dega_files,
        cbg=cbg_custom,
        verbose=True,
        segmentation_approach=segmentation_parameters["segmentation_approach"],
    )

    create_cluster_and_meta_cluster(
        technology=segmentation_parameters["technology"],
        path_dega_files=path_dega_files,
        segmentation_approach=segmentation_parameters["segmentation_approach"],
    )

    # Get the first .dzi file in sorted order
    dzi_files = sorted((Path(path_dega_files) / "pyramid_images").glob("*.dzi"))
    if not dzi_files:
        raise FileNotFoundError("No .dzi files found in pyramid_images.")

    # Use the first .dzi file
    tree = ET.parse(dzi_files[0])
    root = tree.getroot()
    width = int(root[0].attrib["Width"])
    height = int(root[0].attrib["Height"])

    tile_bounds = {"x_min": 0, "x_max": width, "y_min": 0, "y_max": height}

    make_cell_boundary_tiles(
        technology=segmentation_parameters["technology"],
        path_cell_boundaries=str(Path(path_segmentation_files) / "cell_polygons.parquet"),
        path_output=str(
            Path(path_dega_files)
            / f"cell_segmentation_{segmentation_parameters['segmentation_approach']}"
        ),
        tile_size=tile_size,
        tile_bounds=tile_bounds,
        image_scale=image_scale,
    )

    cluster_gene_expression(
        technology=segmentation_parameters["technology"],
        path_dega_files=path_dega_files,
        cbg=cbg_custom,
        segmentation_approach=segmentation_parameters["segmentation_approach"],
    )

    save_landscape_parameters(
        technology=segmentation_parameters["technology"],
        path_dega_files=path_dega_files,
        image_name="dapi_files",
        tile_size=tile_size,
        image_format=".webp",
        segmentation_approach=segmentation_parameters["segmentation_approach"],
    )


__all__ = [
    "_to_geometry",
    "add_clustering_from_adata",
    "boundary_tile",
    "get_image_info",
    "landscape",
    "main",
    "make_trx_tiles",
    "nbhd_cloud",
    "read_cbg_mtx",
    "resolve_xenium_morphology_ome_path",
    "trx_tile",
    "write_cell_clusters_meta",
    "write_gene_cell_scatter",
    "write_gene_shapes",
    "write_gene_shapes_streaming",
    "write_identity_transform",
    "write_meta_gene_for_nbhd_cloud",
    "write_meta_slice",
    "write_nbhd_cloud_cells",
    "write_nbhd_cloud_dataset",
    "write_nbhd_cloud_shapes_and_features",
]
