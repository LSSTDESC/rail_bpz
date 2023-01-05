# rail_bpz
RAIL interface to BPZ algorithms via the [DESC_BPZ](https://github.com/LSSTDESC/DESC_BPZ) package implementation (also available via PyPI with `pip install desc-bpz`).

As the "lite" name implies, not all features of BPZ are implemented, the main product is the marginalized redshift PDF, which is output for a sample as a `qp` ensemble.  However, several other quantities are computed and stored as "ancillary" data and stored with the ensemble, these are:
- zmode (float): the mode of the marginalized posterior redshift PDF distribution.
- zmean (float): the mean of the marginalized posterior redshift PDF distribution.
- tb (int): the integer index for the "best-fit" SED template **at the redshift mode, `zmode`**.  Note that the best-fit template can be different at different redshifts as the SED observed colors change with redshift, so you **can not** assume this single SED for the full marginalized PDF, it should only be used for the "point estimate" redshift `zmode`.
- todds (float): relating to the comment above on tb, `todds` is a new quantity not included with the original BPZ implementation, it is the fraction of marginalized posterior probability assigned to `tb`.  So, high values of `todds` would mean that no other templates fit well, even at other redshifts, while a lower value of `todds` means that there are alternative fits, either at the same redshift or other redshifts.  If you are wanting to compute physical quantities based on `tb`, a lower value of `todds` would mean that such fits would be missing degenerate SED solutions, and should not be trusted.


