# coding=utf-8
# Copyright (c) HISSL Contributors
try:
    import h5py
except ImportError:
    raise ValueError("You must have h5py installed to run this script: pip install h5py.")

from pathlib import Path

# Idea of this script is to take the TCGA-CRCk or TCGA-BC features saved per tile in h5 format and output them
# in a way that is most efficient for any classification model to work with them.

# We don't want to do this in VISSL because
# - the dataset there doesn't load them per patient
# - we don't want to assume that everything can be kept in memory
# - appending to a dataset in h5 is not meant to be done

# So what we do here instead with a small script, is
# - loop over all datasets in a SINGLE h5 file...
# - save all paths
# - get the unique patient IDs and their tiles
# - loop over each patient, place all features from all tiles in a single object
# - save this single object per patient


def compile_tcga_crck(cfg):
    hf = h5py.File(cfg["PATH_TO_INPUT_H5"], "r")

    # Get the principle group. This is a path per tile that will hold the feature vector and all metadata
    tile_groups = []
    hf.visititems(lambda x, y: (tile_groups.append(str(x))) if x.endswith(".png") else None)

    # Get all (unique) slide IDs and labels from the filenames
    slide_ids = [hf[f"{x}/meta/slide_id"][()].decode("UTF-8") for x in tile_groups]
    unique_slide_ids = list(set(slide_ids))

    # Get which tile belongs to which slide
    slide_ids_to_tile_groups = {
        unique_slide_id: [
            tile_group for slide_id, tile_group in zip(slide_ids, tile_groups) if slide_id == unique_slide_id
        ]
        for unique_slide_id in unique_slide_ids
    }

    # For each slide, compile all the data that's required, and save it to a separate h5
    for unique_slide_id in unique_slide_ids:
        slide_output_h5_filename = Path(cfg["PATH_TO_OUTPUT_H5"]).stem + f"_{unique_slide_id}"
        slide_output_h5_filepath = cfg["PATH_TO_INPUT_H5"].replace(
            Path(cfg["PATH_TO_INPUT_H5"]).stem, slide_output_h5_filename
        )
        print(f"Writing to {slide_output_h5_filepath}")
        new_hf = h5py.File(slide_output_h5_filepath, "a")

        all_features = []  # should all be different
        all_vissl_ids = []  # should all be different
        all_slide_ids = []  # should all be the same
        all_case_ids = []  # Should all be the same
        all_targets = []  # should all be the same
        all_paths = []

        for tile_group in slide_ids_to_tile_groups[unique_slide_id]:
            print(tile_group)
            all_features.append(hf[f"{tile_group}/data/heads"][()])
            all_vissl_ids.append(hf[f"{tile_group}/meta/vissl_id"][()])

            all_slide_ids.append(hf[f"{tile_group}/meta/slide_id"][()])
            all_case_ids.append(hf[f"{tile_group}/meta/case_id"][()])
            all_targets.append(1 if "MSIMUT" in tile_group else 0)
            all_paths.append(tile_group.replace(cfg["PATHS_ROOT"], ""))

        assert len(set(all_targets)) == 1
        assert len(set(all_case_ids)) == 1
        assert len(set(all_slide_ids)) == 1

        assert len(set(all_vissl_ids)) == len(all_vissl_ids) == len(all_features)

        new_hf["data"] = all_features
        new_hf["target"] = all_targets[0]
        new_hf["case_id"] = all_case_ids[0]
        new_hf["slide_id"] = all_slide_ids[0]
        new_hf["paths"] = all_paths
        new_hf["root_dir"] = cfg["PATHS_ROOT"]

    return


def compile_tcga_bc(cfg):
    hf = h5py.File(cfg["PATH_TO_INPUT_H5"], "r")

    # Get the principle group. This is a path per tile that will hold the feature vector and all metadata
    wsi_groups = []

    case_keys = hf[cfg["PATHS_ROOT"]].keys()

    for case_key in case_keys:
        wsi_groups += [f"{cfg['PATHS_ROOT']}/{case_key}/{slide_key}" for slide_key in hf[f"{cfg['PATHS_ROOT']}/{case_key}"].keys()]

    # hf.visititems(lambda x, y: (wsi_groups.append(str(x))) if x.endswith(".svs") else None)

    for wsi_group in wsi_groups:

        case_id = wsi_group.split('/')[-2]
        slide_id = wsi_group.split('/')[-1]

        slide_output_h5_filename = slide_id + '.h5'

        slide_output_h5_filepath = cfg["PATH_TO_INPUT_H5"].replace(
            Path(cfg["PATH_TO_INPUT_H5"]).name, (f'{case_id}/' + slide_output_h5_filename)
        )

        print(f"Writing to {slide_output_h5_filepath}")

        Path.mkdir(Path(cfg["PATH_TO_INPUT_H5"]).parent / Path(case_id), exist_ok=True)

        new_hf = h5py.File(slide_output_h5_filepath, "a")

        all_features = []  # should all be different
        all_vissl_ids = []  # should all be different
        all_x = []  # should all be the same
        all_y = []  # Should all be the same
        all_h = []
        all_w = []
        all_mpp = []
        all_region_index = []
        vissl_indices = []

        for vissl_idx in hf[wsi_group].keys():

            all_features.append(hf[f"{wsi_group}/{vissl_idx}/data/heads"][()])
            all_vissl_ids.append(vissl_idx)

            all_x.append(hf[f"{wsi_group}/{vissl_idx}/meta/x"][()])
            all_y.append(hf[f"{wsi_group}/{vissl_idx}/meta/y"][()])
            all_w.append(hf[f"{wsi_group}/{vissl_idx}/meta/w"][()])
            all_h.append(hf[f"{wsi_group}/{vissl_idx}/meta/h"][()])
            all_mpp.append(hf[f"{wsi_group}/{vissl_idx}/meta/mpp"][()])
            all_region_index.append(hf[f"{wsi_group}/{vissl_idx}/meta/region_index"][()])
            vissl_indices.append(vissl_idx)

        new_hf["data"] = all_features
        new_hf["path"] = wsi_group.replace(cfg["PATHS_ROOT"], "")
        new_hf["root_dir"] = cfg["PATHS_ROOT"]
        new_hf["case_id"] = case_id
        new_hf["slide_id"] = slide_id
        new_hf['x'] = all_x
        new_hf['y'] = all_y
        new_hf['w'] = all_w
        new_hf['h'] = all_h
        new_hf['mpp'] = all_mpp
        new_hf['region_index'] = all_region_index
        new_hf['vissl_index'] = vissl_indices

        new_hf.close()

def main(cfg):
    if cfg["DATASET_TYPE"] == "tcga_crck":
        compile_tcga_crck(cfg)
    elif cfg["DATASET_TYPE"] == "tcga_bc":
        compile_tcga_bc(cfg)

if __name__ == "__main__":
    for split in ['train', 'test']:
        cfg = {}
        cfg["DATASET_TYPE"] = "tcga_crck"  # tcga_crck or tcga_bc
        cfg[
            "PATH_TO_INPUT_H5"
        ] = f"/project/schirris/hissl-logs/deepsmile-rev/extract-features/tcga-crck/imagenet/8475262/rank0_{split}_output.hd5"
        cfg["OUTPUT_SUFFIX"] = "compiled"
        cfg["PATHS_ROOT"] = "project/schirris/data/kather_data/data/msidata/crc_dx"
        OUTPUT_H5_FILENAME = Path(cfg["PATH_TO_INPUT_H5"]).stem + f'_{cfg["OUTPUT_SUFFIX"]}'
        cfg["PATH_TO_OUTPUT_H5"] = cfg["PATH_TO_INPUT_H5"].replace(Path(cfg["PATH_TO_INPUT_H5"]).stem, OUTPUT_H5_FILENAME)
        main(cfg)

    # RUN FOR TCGA-BC
    # rootdir = "/project/schirris/hissl-logs/deepsmile-rev/extract-features/tcga-bc/"
    # reldirs = ['batch-1024/epoch60/fold4/8469158', 'imagenet/8469160']
    # filename = "rank0_train_output.hd5"
    # cfg = {}
    # for reldir in reldirs:
    #
    #     cfg["DATASET_TYPE"] = "tcga_bc"  # tcga_crck or tcga_bc
    #     cfg["PATHS_ROOT"] = "project/schirris/data/tcga_brca_dx/data_large/gdc_manifest_all_BRCA_DX-2020-08-05/images"
    #     cfg["PATH_TO_INPUT_H5"] = str(Path(rootdir) / Path(reldir) / Path(filename))
    #
    #     cfg["OUTPUT_SUFFIX"] = "compiled"
    #     OUTPUT_H5_FILENAME = Path(cfg["PATH_TO_INPUT_H5"]).stem + f'_{cfg["OUTPUT_SUFFIX"]}'
    #     cfg["PATH_TO_OUTPUT_H5"] = cfg["PATH_TO_INPUT_H5"].replace(Path(cfg["PATH_TO_INPUT_H5"]).stem, OUTPUT_H5_FILENAME)
    #
    #
    #     print(f'Reading {cfg["PATH_TO_INPUT_H5"]}')
    #     print(f'Writing to {cfg["PATH_TO_OUTPUT_H5"]}')
    #     print(f'Compiling for {cfg["DATASET_TYPE"]}')
    #
    #     main(cfg)
