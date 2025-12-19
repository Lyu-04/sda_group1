import xarray as xr
import pandas as pd


file = "dataset1986-2021.grib"

#  Instantaneous fields (the ones you already have)
ds_inst = xr.open_dataset(
    file,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"stepType": "instant"}}
)
# This should contain: u10, v10, d2m, t2m, msl, tcc

# Total precipitation (tp)
ds_tp = xr.open_dataset(
    file,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "tp"}}
)

#  Evaporation (e)
ds_e = xr.open_dataset(
    file,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "e"}}
)

#  Potential evaporation (pev)
ds_pev = xr.open_dataset(
    file,
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"shortName": "pev"}}
)

# 🔧 Sometimes these have 'valid_time' instead of 'time' – align names:
for ds in (ds_tp, ds_e, ds_pev):
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

#  Align them all to the same time/lat/lon grid as ds_inst
ds_tp  = ds_tp.reindex_like(ds_inst, method="nearest")
ds_e   = ds_e.reindex_like(ds_inst, method="nearest")
ds_pev = ds_pev.reindex_like(ds_inst, method="nearest")

#  Merge everything into one Dataset
ds_all = xr.merge(
    [ds_inst, ds_tp[["tp"]], ds_e[["e"]], ds_pev[["pev"]]],
    compat="override"
)

#  To CSV
df = ds_all.to_dataframe().reset_index()
df.to_csv("grid_all_vars.csv", index=False)

'''# open GRIB file
ds = xr.open_dataset("ef4f7494403816216860ae3d5e018f05.grib", engine="cfgrib")

# convert to dataframe
df = ds.to_dataframe().reset_index()

# save as CSV
df.to_csv("output_with_coords1.csv", index=False)
'''

