import os
import pickle

import numpy as np
import pytest
import scipy.special
import tables_io
from rail.core.data import DataStore, TableHandle
from rail.core.stage import RailStage
from rail.utils.path_utils import RAILDIR
from rail.utils.testing_utils import one_algo

from rail.bpz.utils import RAIL_BPZ_DIR
from rail.estimation.algos import bpz_lite

sci_ver_str = scipy.__version__.split(".")

parquetdata = "./tests/validation_10gal.pq"
fitsdata = "./tests/validation_10gal.fits"
traindata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
validdata = os.path.join(RAILDIR, "rail/examples_data/testdata/validation_10gal.hdf5")

# DS = RailStage.data_store
# DS.__class__.allow_overwrite = True


@pytest.mark.parametrize(
    "ntarray, inputdata, groupname, size",
    [
        ([8], parquetdata, "", 10),
        ([4, 4], traindata, "photometry", 100),
        ([8], fitsdata, "", 10),
    ],
)
def test_bpz_train(ntarray, inputdata, groupname, size):
    # first, train with two broad types
    train_config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "hdf5_groupname": groupname,
        "nt_array": ntarray,
        "type_file": "tmp_broad_types.hdf5",
        "model": "testmodel_bpz.pkl",
        "output_hdfn": False,
    }
    if len(ntarray) == 2:
        broad_types = np.random.randint(2, size=size)
    else:
        broad_types = np.zeros(size, dtype=int)
    typedict = dict(types=broad_types)
    tables_io.write(typedict, "tmp_broad_types.hdf5")
    train_algo = bpz_lite.BPZliteInformer
    # DS.clear()
    # training_data = DS.read_file("training_data", TableHandle, inputdata)
    training_data = TableHandle("training_data", path=inputdata)
    training_data = training_data.read()
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)
    expected_keys = ["fo_arr", "kt_arr", "zo_arr", "km_arr", "a_arr", "mo", "nt_array"]
    with open("testmodel_bpz.pkl", "rb") as f:
        tmpmodel = pickle.load(f)
    for key in expected_keys:
        assert key in tmpmodel.keys()
    os.remove("tmp_broad_types.hdf5")


def test_output_hdfn_inform():
    train_config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "hdf5_groupname": "photometry",
        "nt_array": [1, 2, 5],
        "type_file": "tmp_broad_types.hdf5",
        "model": "testmodel_bpz.pkl",
        "output_hdfn": True,
    }
    train_algo = bpz_lite.BPZliteInformer
    # DS.clear()
    # training_data = DS.read_file("training_data", TableHandle, traindata)
    training_data = TableHandle("training_data", path=traindata)
    training_data = training_data.read()
    train_stage = train_algo.make_stage(**train_config_dict)
    train_stage.inform(training_data)
    expected_keys = ["fo_arr", "kt_arr", "zo_arr", "km_arr", "a_arr", "mo", "nt_array"]
    with open("testmodel_bpz.pkl", "rb") as f:
        tmpmodel = pickle.load(f)
    for key in expected_keys:
        assert key in tmpmodel.keys()


def test_bpz_lite():
    train_config_dict = {}
    estim_config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "nzbins": 301,
        "bpz_ref_data_path": None,
        "columns_file": os.path.join(
            RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"
        ),
        "spectra_file": "CWWSB4.list",
        "madau_flag": "no",
        "no_prior": False,
        "ref_band": "mag_i_lsst",
        "prior_file": "hdfn_gen",
        "p_min": 0.005,
        "gauss_kernel": 0.0,
        "zp_errors": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        "mag_err_min": 0.005,
        "hdf5_groupname": "photometry",
        "nt_array": [8],
        "model": "testmodel_bpz.pkl",
    }
    zb_expected = np.array([0.16, 0.12, 0.0, 0.12, 0.05, 0.14, 0.11, 0.14, 0.05, 0.16])
    train_algo = None
    pz_algo = bpz_lite.BPZliteEstimator
    results, rerun_results, rerun3_results = one_algo(
        "BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    # assert np.isclose(results.ancil['zmode'], zb_expected, atol=0.03).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()


@pytest.mark.parametrize(
    "inputdata, groupname",
    [
        (parquetdata, ""),
        (validdata, "photometry"),
        (fitsdata, ""),
    ],
)
def test_bpz_wHDFN_prior(inputdata, groupname):
    estim_config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "nzbins": 301,
        "bpz_ref_data_path": None,
        "columns_file": os.path.join(
            RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"
        ),
        "spectra_file": "CWWSB4.list",
        "madau_flag": "no",
        "ref_band": "mag_i_lsst",
        "prior_file": "flat",
        "p_min": 0.005,
        "gauss_kernel": 0.1,
        "zp_errors": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        "mag_err_min": 0.005,
        "hdf5_groupname": groupname,
        "nt_array": [1, 2, 5],
        "model": os.path.join(
            RAILDIR, "rail/examples_data/estimation_data/data/CWW_HDFN_prior.pkl"
        ),
    }
    zb_expected = np.array([0.18, 2.88, 0.14, 0.19, 2.91, 0.18, 0.21, 0.21, 2.98, 2.92])

    # validation_data = DS.read_file("validation_data", TableHandle, inputdata)
    validation_data = TableHandle("validation_data", path=inputdata)
    validation_data = validation_data.read()
    pz = bpz_lite.BPZliteEstimator.make_stage(name="bpz_hdfn", **estim_config_dict)
    results = pz.estimate(validation_data)
    assert np.isclose(results.data.ancil["zmode"], zb_expected, atol=0.05).all()
    # DS.clear()
    os.remove(pz.get_output(pz.get_aliased_tag("output"), final_name=True))


def test_wrong_number_of_filters():
    train_config_dict = {}
    estim_config_dict = {
        "zmin": 0.0,
        "zmax": 3.0,
        "dz": 0.01,
        "nzbins": 301,
        "bpz_ref_data_path": None,
        "columns_file": os.path.join(
            RAIL_BPZ_DIR, "rail/examples_data/estimation_data/configs/test_bpz.columns"
        ),
        "spectra_file": "CWWSB4.list",
        "madau_flag": "no",
        "ref_band": "mag_i_lsst",
        "prior_file": "flat",
        "p_min": 0.005,
        "gauss_kernel": 0.1,
        "zp_errors": np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        "mag_err_min": 0.005,
        "filter_list": ["DC2LSST_u", "DC2LSST_g"],
        "hdf5_groupname": "photometry",
    }
    train_algo = None
    with pytest.raises(ValueError):
        pz_algo = bpz_lite.BPZliteEstimator
        _, _, _ = one_algo(
            "BPZ_lite", train_algo, pz_algo, train_config_dict, estim_config_dict
        )
