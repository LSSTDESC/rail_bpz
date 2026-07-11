"""
Port of *some* parts of BPZ, not the entire codebase.
Much of the code is directly ported from BPZ, written
by Txitxo Benitez and Dan Coe (Benitez 2000), which
was modified by Will Hartley and Sam Schmidt to make
it python3 compatible.  It was then modified to work
with TXPipe and ceci by Joe Zuntz and Sam Schmidt
for BPZPipe.  This version for RAIL removes a few
features and concentrates on just predicting the PDF.

Missing from full BPZ:
-no chi^2, ML quantities
-plotting utilities
-no output of 2D probs (maybe later add back in)
-no 'cluster' prior mods

"""

import glob
import os

import numpy as np
import tables_io
from ceci.config import StageParameter as Param
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import Hdf5Handle
from rail.estimation.estimator import CatEstimator
from rail.utils.path_utils import RAILDIR


default_offset_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class BPZlitePreEstimator(CatEstimator):
    """This is a 'pre-estimate' stage that does two things:
    1: compute the zero-point offsets for a training/test set
    2: computes the best "broad type" for a training set that can be used to train
    new prior params in the BPZliteInformer stage
    """

    name = "BPZlitePreEstimator"
    entrypoint_function = "estimate"  # the user-facing science function for this class
    interactive_function = "bpz_lite_preestimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,
        bands=SHARED_PARAMS,
        err_bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        bpz_ref_data_path=Param(
            str,
            "None",
            msg="bpz_ref_data_path (str): file path to the "
            "SED, FILTER, and AB directories.  If left to "
            "default `None` it will use the install "
            "directory for rail + rail/examples_data/estimation_data/data",
        ),
        spectra_file=Param(
            str,
            "CWWSB4.list",
            msg="name of the file specifying the list of SEDs to use",
        ),
        m0=Param(float, 20.0, msg="reference apparent mag, used in prior param"),
        nt_array=Param(
            list,
            [1, 2, 5],
            msg="list of integer number of templates per 'broad type', "
            "must be in same order as the template set, and must sum to the same number "
            "as the # of templates in the spectra file",
        ),
        mmin=Param(
            float, 18.0, msg="lowest apparent mag in ref band, lower values ignored"
        ),
        mmax=Param(
            float, 29.0, msg="highest apparent mag in ref band, higher values ignored"
        ),
        dz=Param(float, 0.01, msg="delta z in grid"),
        unobserved_val=Param(
            float,
            -99.0,
            msg="value to be replaced with zero flux and given large errors for non-observed filters",
        ),
        filter_list=SHARED_PARAMS,
        madau_flag=Param(
            str,
            "no",
            msg="set to 'yes' or 'no' to set whether to include intergalactic "
            "Madau reddening when constructing model fluxes",
        ),
        no_prior=Param(bool, False, msg="set to True if you want to run with no prior"),
        p_min=Param(
            float,
            0.005,
            msg="BPZ sets all values of "
            "the PDF that are below p_min*peak_value to 0.0, "
            "p_min controls that fractional cutoff",
        ),
        gauss_kernel=Param(
            float,
            0.0,
            msg="gauss_kernel (float): BPZ "
            "convolves the PDF with a kernel if this is set "
            "to a non-zero number",
        ),
        zp_errors=SHARED_PARAMS,
        mag_err_min=Param(
            float,
            0.005,
            msg="a minimum floor for the magnitude errors to prevent a "
            "large chi^2 for very very bright objects",
        ),
        only_type=Param(
            bool,
            True,
            msg="if set to True, fixes the redshift to be true z from "
            "redshift_col for computing best broad type and offsets"
        ),
        zp_offsets=Param(
            list,
            default_offset_array,
            msg="zero point offsets to apply (if doing iteratively for type fit)"
        ),
    )
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Init function, init config stuff"""
        super().__init__(args, **kwargs)
        self.fo_arr = None
        self.kt_arr = None
        self.typmask = None
        self.ntyp = None
        self.mags = None
        self.szs = None
        self.besttypes = None
        self.best_broadtypes = None
        self.m0 = self.config.m0
        self.config.zp_offsets = np.array(self.config.zp_offsets)

        datapath = self.config["bpz_ref_data_path"]
        if datapath is None or datapath == "None":
            tmpdatapath = os.path.join(
                RAILDIR, "rail/examples_data/estimation_data/data"
            )
            os.environ["BPZDATAPATH"] = tmpdatapath
            self.bpz_ref_data_path = tmpdatapath
        else:  # pragma: no cover
            self.bpz_ref_data_path = datapath
            os.environ["BPZDATAPATH"] = self.bpz_ref_data_path
        if not os.path.exists(self.bpz_ref_data_path):  # pragma: no cover
            raise FileNotFoundError(
                "BPZDATAPATH "
                + self.bpz_ref_data_path
                + " does not exist! Check value of bpz_ref_data_path in config file!"
            )

        self.flux_templates = self._load_templates()

    def _load_templates(self):
        from desc_bpz.useful_py3 import get_data, get_str, match_resol

        # The redshift range we will evaluate on
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        z = self.zgrid

        bpz_ref_data_path = self.bpz_ref_data_path
        filters = self.config.filter_list

        spectra_file = os.path.join(bpz_ref_data_path, "SED", self.config.spectra_file)
        spectra = [s[:-4] for s in get_str(spectra_file)]

        nt = len(spectra)
        nf = len(filters)
        nz = len(z)
        flux_templates = np.zeros((nz, nt, nf))

        ab_dir = os.path.join(bpz_ref_data_path, "AB")
        os.makedirs(ab_dir, exist_ok=True)

        # make a list of all available AB files in the AB directory
        ab_file_list = glob.glob(ab_dir + "/*.AB")
        ab_file_db = [os.path.split(x)[-1] for x in ab_file_list]

        for i, s in enumerate(spectra):
            for j, f in enumerate(filters):
                if self.config.madau_flag == "yes":
                    mflag = "withmadau"
                else:
                    mflag = "nomadau"
                model = f"{s}.{f}.{mflag}.AB"
                if model not in ab_file_db:  # pragma: no cover
                    self._make_new_ab_file(s, f, mflag)
                model_path = os.path.join(bpz_ref_data_path, "AB", model)
                zo, f_mod_0 = get_data(model_path, (0, 1))
                flux_templates[:, i, j] = match_resol(zo, f_mod_0, z)

        return flux_templates

    def _make_new_ab_file(self, spectrum, filter_, mflag):  # pragma: no cover
        from desc_bpz.bpz_tools_py3 import ABflux

        new_file = f"{spectrum}.{filter_}.{mflag}.AB"
        self.log.info(f"  Generating new AB file {new_file}....")
        ABflux(spectrum, filter_, self.config.madau_flag)

    def _preprocess_magnitudes(self, data):
        from desc_bpz.bpz_tools_py3 import e_mag2frac

        bands = self.config.bands
        errs = self.config.err_bands

        fluxdict = {}

        # Load the magnitudes
        zp_frac = e_mag2frac(np.array(self.config.zp_errors))

        # replace non-detects with 99 and mag_err with lim_mag for consistency
        # with typical BPZ performance
        for ii, (bandname, errname) in enumerate(zip(bands, errs)):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = np.isnan(data[bandname])
            else:
                detmask = np.isclose(data[bandname], self.config.nondetect_val)
            data[bandname] -= self.config.zp_offsets[ii]
            data[bandname][detmask] = 99.0
            data[errname][detmask] = self.config.mag_limits[bandname]

        # replace non-observations with -99, again to match BPZ standard
        # below the fluxes for these will be set to zero but with enormous
        # flux errors
        for bandname, errname in zip(bands, errs):
            if np.isnan(self.config.unobserved_val):  # pragma: no cover
                obsmask = np.isnan(data[bandname])
            else:
                obsmask = np.isclose(data[bandname], self.config.unobserved_val)
            data[bandname][obsmask] = -99.0
            data[errname][obsmask] = 20.0

        # Only one set of mag errors
        mag_errs = np.array([data[er] for er in errs]).T

        # Group the magnitudes and errors into one big array
        mags = np.array([data[b] for b in bands]).T

        # Clip to min mag errors.
        # JZ: Changed the max value here to 20 as values in the lensfit
        # catalog of ~ 200 were causing underflows below that turned into
        # zero errors on the fluxes and then nans in the output
        np.clip(mag_errs, self.config.mag_err_min, 20, mag_errs)

        # Convert to pseudo-fluxes
        flux = 10.0 ** (-0.4 * mags)
        flux_err = flux * (10.0 ** (0.4 * mag_errs) - 1.0)

        # Check if an object is seen in each band at all.
        # Fluxes not seen at all are listed as infinity in the input,
        # so will come out as zero flux and zero flux_err.
        # Check which is which here, to use with the ZP errors below
        seen1 = (flux > 0) & (flux_err > 0)
        seen = np.where(seen1)
        # unseen = np.where(~seen1)
        # replace Joe's definition with more standard BPZ style
        nondetect = 99.0
        nondetflux = 10.0 ** (-0.4 * nondetect)
        unseen = np.isclose(flux, nondetflux, atol=nondetflux * 0.5)

        # replace mag = 99 values with 0 flux and 1 sigma limiting magnitude
        # value, which is stored in the mag_errs column for non-detects
        # NOTE: We should check that this same convention will be used in
        # LSST, or change how we handle non-detects here!
        flux[unseen] = 0.0
        flux_err[unseen] = 10.0 ** (-0.4 * np.abs(mag_errs[unseen]))

        # Add zero point magnitude errors.
        # In the case that the object is detected, this
        # correction depends onthe flux.  If it is not detected
        # then BPZ uses half the errors instead
        add_err = np.zeros_like(flux_err)
        add_err[seen] = ((zp_frac * flux) ** 2)[seen]
        add_err[unseen] = ((zp_frac * 0.5 * flux_err) ** 2)[unseen]
        flux_err = np.sqrt(flux_err**2 + add_err)

        # Convert non-observed objects to have zero flux
        # and enormous error, so that their likelihood will be
        # flat. This follows what's done in the bpz script.
        nonobserved = -99.0
        unobserved = np.isclose(mags, nonobserved)
        flux[unobserved] = 0.0
        flux_err[unobserved] = 1e30

        # Upate the flux dictionary with new things we have calculated
        fluxdict["flux"] = flux
        fluxdict["flux_err"] = flux_err
        m_0_col = self.config.bands.index(self.config.ref_band)
        fluxdict["mag0"] = mags[:, m_0_col]

        return fluxdict

    def _types_ratios(self, flux_templates, flux, flux_err, mag_0, z, sz=None):
        from desc_bpz.bpz_tools_py3 import p_c_z_t
        from desc_bpz.prior_from_dict import prior_function

        modeldict = self.modeldict
        p_min = self.config.p_min
        nt = flux_templates.shape[1]

        # The likelihood and prior...
        pczt = p_c_z_t(flux, flux_err, flux_templates)
        L = pczt.likelihood

        if self.config.only_type:
            # fix to only include the true z row
            whichrow = np.searchsorted(z, sz)
            # set all rows of prior to zero except the specz row
            # and set "flat" prior for that row
            P = np.zeros(L.shape)
            P[whichrow, :] = 1. / nt
        else:

            # old prior code returns NoneType for prior if "flat" or "none"
            # just hard code the no prior case for now for backward compatibility
            if self.config.no_prior:  # pragma: no cover
                P = np.ones(L.shape)
            else:
                # set num templates to nt, which is hardcoding to "interp=0"
                # in BPZ, i.e. do not create any interpolated templates
                P = prior_function(z, mag_0, modeldict, nt)

        post = L * P
        # Right now we jave the joint PDF of p(z,template). Marginalize
        # over the templates to just get p(z)
        post_z = post.sum(axis=1)

        # Find the mode
        zpos = np.argmax(post_z)
        zmode = self.zgrid[zpos]

        # Trim probabilities
        # below a certain threshold pct of p_max
        p_max = post_z.max()
        post_z[post_z < (p_max * p_min)] = 0

        # Normalize in the same way that BPZ does
        # But, only normalize if the elements don't sum to zero
        # if they are all zero, just leave p(z) as all zeros, as no templates
        # are a good fit.
        if not np.isclose(post_z.sum(), 0.0):
            post_z /= post_z.sum()

        # Find T_B, the highest probability template *at zmode*
        tmode = post[zpos, :]
        t_b = np.argmax(tmode)

        ft = flux_templates[zpos, t_b, :]

        # frat = flux/ft
        # fw = flux/flux_err

        return zmode, t_b, ft

    def run(self):
        self.refcol = self.config.bands.index(self.config.ref_band)
        self.nt_array = self.config.nt_array
        if not self.config.no_prior:
            # the parameters for the HDFN prior
            self.fo_arr = np.array([0.35, 0.5])
            self.kt_arr = np.array([0.45, 0.147])
            self.zo_arr = np.array([0.431, 0.39, 0.0626])
            self.km_arr = np.array([0.0913, 0.0636, 0.123])
            self.a_arr = np.array([2.465, 1.806, 0.906])
            self.m0 = 20.0
            self.nt_array = self.config.nt_array
            self.modeldict = dict(
                fo_arr=self.fo_arr,
                kt_arr=self.kt_arr,
                zo_arr=self.zo_arr,
                km_arr=self.km_arr,
                a_arr=self.a_arr,
                mo=self.m0,
                nt_array=self.nt_array
            )
        else:
            self.m0 = self.config.m0
            self.modeldict = dict(fakeparam=0.0)  # not actually used in this case

        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        # convert training data format to numpy dictionary
        if tables_io.types.table_type(training_data) != 1:  # pragma: no cover
            training_data = self._convert_table_format(
                training_data, out_fmt_str="numpyDict"
            )

        # ngal = len(training_data[self.config.ref_band])

        if self.config.ref_band not in training_data.keys():  # pragma: no cover
            raise KeyError(
                f"ref_band {self.config.ref_band} not found in input data!"
            )
        if self.config.redshift_col not in training_data.keys():  # pragma: no cover
            raise KeyError(
                f"redshift column {self.config.redshift_col} not found in input data!"
            )

        #
        test_data = self._preprocess_magnitudes(training_data)
        # m_0_col = self.config.bands.index(self.config.ref_band)

        # nz = len(self.zgrid)
        ng = test_data["flux"].shape[0]
        nf = len(self.config.filter_list)

        zmode = np.zeros(ng)
        ft = np.zeros([ng, nf])
        tb = np.zeros(ng)
        # broad_type = np.zeros(ng)
        flux_temps = self.flux_templates
        zgrid = self.zgrid
        # Loop over all ng galaxies!
        for i in range(ng):
            mag_0 = test_data["mag0"][i]
            flux = test_data["flux"][i]
            flux_err = test_data["flux_err"][i]
            if self.config.only_type:
                truez = training_data[self.config.redshift_col][i]
            else:
                truez = None
            zmode[i], tb[i], ft[i, :] = self._types_ratios(
                flux_temps, flux, flux_err, mag_0, zgrid, truez
            )
            # normalize the ftSrefcol
            # after removing zero values
            ftthresh = 1.e-50
            ftmask = ft[self.refcol] < ftthresh
            ft[self.refcol][ftmask] = ftthresh
            ft = ft * flux[self.refcol] / ft[self.refcol]

        frat = test_data['flux'] / ft
        # fw = test_data['flux'] / test_data['flux_err']

        offsets = np.zeros(nf)
        totoffsets = np.zeros(nf)

        for i in range(nf):
            fmask = np.logical_and(np.isfinite(frat[:, i]), frat[:, i] > 1.e-38)
            xfr = frat[:, i][fmask]
            offsets[i] = -2.5 * np.log10(np.mean(xfr))

        # set so that zero offset for ref col
        norm_offset = offsets[self.refcol]
        offsets -= norm_offset

        # Need to add offsets from config to get total offsets in case
        # this was done iteratively
        for i in range(nf):
            totoffsets[i] = offsets[i] + self.config.zp_offsets[i]

        # find broad types
        xedges = np.concatenate([np.array([0]), self.nt_array])
        edges = np.zeros(len(xedges), dtype=float)
        for ii in range(len(xedges)):
            edges[ii] = np.sum(xedges[:ii + 1])
        edges -= 0.005
        # find broad types, subtract 1 so they are zero-indexed
        btype = np.searchsorted(edges, tb, side='left') - 1

        outdict = dict(
            offsets=dict(
                zp_offsets=np.array(totoffsets)),
            types=dict(broad_type=np.array(btype),
                       tb=np.array(tb))
        )
        self.add_data("output", outdict)
