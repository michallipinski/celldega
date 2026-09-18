"""Morphology image tiling: DeepZoom pyramids and the format conversions feeding them.

This is the entire pyvips / scikit-image / tifffile surface of ``celldega.pre``, collected
in one module so the optional ``pre`` extra is a per-module concern rather than something
every importer of the package pays for.

pyvips is guarded because it is an optional dependency (``pip install celldega[pre]``);
scikit-image is not, because it currently arrives as a hard transitive dependency of
spatialdata / spatialdata-io / ome-zarr.
"""

from pathlib import Path

from skimage.io import imread, imsave
import tifffile

from .image_info import resolve_xenium_morphology_ome_path


try:
    import pyvips
except ImportError:
    pyvips = None


def _process_image_channel(path_dega_files, channel_info, img):
    """
    Process a single image channel for tiling.

    Parameters:
    - path_dega_files: Landscape files path
    - channel_info: Dictionary with channel information (name, index)
    - img: Optional pre-loaded image array

    Returns:
    - None
    """
    channel_name = channel_info["name"]
    channel_index = channel_info.get("index", 0)

    print(f"generating {channel_name} image tiles ...")

    pyramid_path = Path(path_dega_files) / "pyramid_images" / f"{channel_name}_files"
    if pyramid_path.exists():
        return

    # Extract and process the channel
    scale = 1 if channel_name.lower() == "dapi" else 2  # Adjust intensity for better visualization
    if img.ndim == 3:
        image_data = img[..., channel_index] * scale
    elif img.ndim == 2:
        image_data = img * scale
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}. Expected 2D or 3D image.")

    output_path = Path(path_dega_files) / f"{channel_name}_output_regular.tif"
    imsave(output_path, image_data)

    # Convert the image to PNG format
    image_png = _convert_to_png(str(output_path))

    # Create a DeepZoom pyramid for the channel
    make_deepzoom_pyramid(
        image_png,
        str(Path(path_dega_files) / "pyramid_images"),
        channel_name,
        suffix=".webp[Q=100]",
    )


def create_image_tiles(technology, data_dir, path_dega_files, image_tile_layer="dapi"):
    """
    Creates image tiles for visualization from the Xenium morphology image.

    Args:
        technology (str): The technology used (e.g., "Xenium", "MERSCOPE", "VisiumHD", "H&E").
        data_dir (str): Path to the directory containing the data (e.g., morphology_focus_0000.ome.tif).
        path_dega_files (str): Path to the directory where the image tiles and pyramid will be saved.
        image_tile_layer (str, optional): Specifies which image layers to process. Options for Xenium are
        'dapi' (default) or 'all'. Use the filename of the .scn file for h&e Landscapes.

    Raises:
        ValueError: If the specified technology is not supported or if the image_tile_layer is invalid.
        FileNotFoundError: If the required input image file is not found.
    """
    print("\n========Generating image tiles========")
    if technology == "Xenium":
        print("------ xenium")
        create_image_tiles_xenium(data_dir, path_dega_files, image_tile_layer=image_tile_layer)
    elif technology == "MERSCOPE":
        print("------ merscope")
        create_image_tiles_merscope(data_dir, path_dega_files, image_tile_layer=image_tile_layer)
    elif technology == "h&e":
        print("------ h&e")
        create_image_tiles_h_and_e(data_dir, path_dega_files, image_tile_layer=image_tile_layer)

    print("Image tiles created successfully.")


def create_image_tiles_h_and_e(data_dir, path_dega_files, image_tile_layer):
    """
    Creates image tiles for visualization from the H&E image.

    Args:
        data_dir (str): Path to the directory containing the data (e.g., morphology_focus_0000.ome.tif).
        path_dega_files (str): Path to the directory where the image tiles and pyramid will be saved.
        image_tile_layer (str, optional): Specifies the name of the h&e image to process.
    Raises:
        FileNotFoundError: If the required input image file is not found.
    """
    with tifffile.TiffFile(Path(data_dir) / image_tile_layer) as tif:
        print(tif.pages)  # Show available pages
        image = tif.pages[0].asarray()

        # make this directory, path_dega_files, if it does not exist
        landscape_path = Path(path_dega_files)
        landscape_path.mkdir(exist_ok=True)

        temp_tiff_path = landscape_path / image_tile_layer.replace(".scn", "_output_regular.tif")
        tifffile.imwrite(temp_tiff_path, image)

        # Convert the image to PNG format
        image_png = _convert_to_png(str(temp_tiff_path))

        # Create a DeepZoom pyramid for the DAPI channel
        make_deepzoom_pyramid(
            image_png,
            str(landscape_path / "pyramid_images"),
            "h_and_e",
            suffix=".webp[Q=100]",
        )

        remove_intermediate_files(path_dega_files)


def remove_intermediate_files(path_dega_files):
    """
    Remove intermediate image files.

    Parameters:
    - path_dega_files: Path to landscape files directory
    """
    # Remove intermediate files
    intermediate_image_files = list(Path(path_dega_files).glob("*output_regular*"))
    for file in intermediate_image_files:
        file.unlink()


def create_image_tiles_xenium(data_dir, path_dega_files, image_tile_layer="dapi"):
    """
    Creates image tiles for visualization from the Xenium morphology image.

    Args:
        data_dir (str): Path to the directory containing the data (e.g., morphology_focus_0000.ome.tif).
        path_dega_files (str): Path to the directory where the image tiles and pyramid will be saved.
        image_tile_layer (str, optional): Specifies which image layers to process. Options are 'dapi' (default) or 'all'.
    Raises:
        FileNotFoundError: If the required input image file is not found.
    """
    if image_tile_layer not in ["dapi", "all"]:
        raise ValueError(f"Invalid image_tile_layer: {image_tile_layer}. Must be 'dapi' or 'all'.")

    file_path = resolve_xenium_morphology_ome_path(data_dir)

    # Load the morphology image once if processing multiple channels
    img = imread(file_path)

    if image_tile_layer == "all" and file_path.name == "morphology.ome.tif":
        raise ValueError(
            "image_tile_layer='all' needs a multi-channel morphology_focus OME-TIFF; "
            "this bundle only has morphology.ome.tif. Use image_tile_layer='dapi' or "
            "supply morphology_focus/*.ome.tif from the instrument output."
        )

    # Process the DAPI channel
    if image_tile_layer in ["dapi", "all"]:
        _process_image_channel(path_dega_files, {"name": "dapi", "index": 0}, img)

    # Process additional channels if image_tile_layer is 'all'
    if image_tile_layer == "all":
        for idx, channel in enumerate(["bound", "rna", "prot"]):
            _process_image_channel(path_dega_files, {"name": channel, "index": idx + 1}, img)

    remove_intermediate_files(path_dega_files)


def create_image_tiles_merscope(data_dir, path_dega_files, image_tile_layer="dapi"):
    """
    Creates image tiles for visualization from the Xenium morphology image.

    Args:
        data_dir (str): Path to the directory containing the data (e.g., morphology_focus_0000.ome.tif).
        path_dega_files (str): Path to the directory where the image tiles and pyramid will be saved.
        image_tile_layer (str, optional): Specifies which image layers to process. Options are 'dapi' (default) or 'all'.
    Raises:
        FileNotFoundError: If the required input image file is not found.
    """
    if image_tile_layer not in ["dapi", "all"]:
        raise ValueError(f"Invalid image_tile_layer: {image_tile_layer}. Must be 'dapi' or 'all'.")

    # Define the path to the DAPI image
    dapi_file_path = Path(data_dir) / "images" / "mosaic_DAPI_z3.tif"

    # Check if the DAPI image exists
    if not dapi_file_path.exists():
        raise FileNotFoundError(
            f"The file 'mosaic_DAPI_z3.tif' does not exist in directory '{data_dir}'."
        )

    # Load the DAPI image once if processing multiple channels
    img_dapi = imread(dapi_file_path)

    # Process the DAPI channel
    _process_image_channel(path_dega_files, {"name": "dapi", "index": 0}, img_dapi)

    # Process additional channels if image_tile_layer is 'all'
    if image_tile_layer == "all":
        # Define the path to the boundary image
        bounda_file_path = Path(data_dir) / "images" / "mosaic_Cellbound1_z3.tif"

        # Check if the boundary image exists
        if not bounda_file_path.exists():
            raise FileNotFoundError(
                f"The file 'mosaic_Cellbound1_z3.tif' does not exist in directory '{data_dir}'."
            )

        # Load the boundary image once if processing multiple channels
        img_bound = imread(bounda_file_path)

        # Process the boundary channel
        _process_image_channel(path_dega_files, {"name": "bound", "index": 0}, img_bound)

    remove_intermediate_files(path_dega_files)


def _reduce_image_size(image_path, scale_image=0.5, path_dega_files=""):
    """Reduces the size of an image by a specified scale factor.

    Args:
        image_path (str): Path to the image file.
        scale_image (float, optional): Scale factor for the image resize. Defaults to 0.5.
        path_dega_files (str, optional): Directory to save the resized image. Defaults to "".

    Returns:
        str: Path to the resized image file.
    """
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    resized_image = image.resize(scale_image)

    new_image_name = Path(image_path).name.replace(".tif", "_downsize.tif")
    new_image_path = Path(path_dega_files) / new_image_name
    resized_image.write_to_file(str(new_image_path))

    return str(new_image_path)


def _convert_to_jpeg(image_path, quality=80):
    """Converts a TIFF image to a JPEG image with a specified quality score.

    Args:
        image_path (str): Path to the image file.
        quality (int, optional): Quality score for the JPEG image. Defaults to 80.

    Returns:
        str: Path to the JPEG image file.
    """
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    new_image_path = str(Path(image_path).with_suffix(".jpeg"))
    image.jpegsave(new_image_path, Q=quality)

    return new_image_path


def _convert_to_png(image_path):
    """Converts a TIFF image to a PNG image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Path to the PNG image file.
    """
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    new_image_path = str(Path(image_path).with_suffix(".png"))
    image.pngsave(new_image_path)

    return new_image_path


def _convert_to_webp(image_path, quality=100):
    """Converts a TIFF image to a WEBP image with a specified quality score.

    Args:
        image_path (str): Path to the image file.
        quality (int, optional): Quality score for the WEBP image. Defaults to 100.

    Returns:
        str: Path to the WEBP image file.
    """
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    new_image_path = str(Path(image_path).with_suffix(".webp"))
    image.webpsave(new_image_path, Q=quality)

    return new_image_path


def make_deepzoom_pyramid(
    image_path, output_path, pyramid_name, tile_size=512, overlap=0, suffix=".jpeg"
):
    """Creates a DeepZoom image pyramid from a JPEG image.

    Args:
        image_path (str): Path to the JPEG image file.
        output_path (str): Directory to save the DeepZoom pyramid.
        pyramid_name (str): Name of the pyramid directory.
        tile_size (int, optional): Tile size for the DeepZoom pyramid. Defaults to 512.
        overlap (int, optional): Overlap size for the DeepZoom pyramid. Defaults to 0.
        suffix (str, optional): Suffix for the DeepZoom pyramid tiles. Defaults to ".jpeg".

    Returns:
        None
    """
    output_path = Path(output_path)
    image = pyvips.Image.new_from_file(image_path, access="sequential")
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / pyramid_name
    image.dzsave(str(output_path), tile_size=tile_size, overlap=overlap, suffix=suffix)
