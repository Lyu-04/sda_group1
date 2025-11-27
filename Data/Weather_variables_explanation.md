Weather variables (from the ERA-style grid dataset)

latitude – Grid point latitude (degrees).

longitude – Grid point longitude (degrees).

u10 – 10-meter u-component of wind: east–west wind. Positive = wind blowing eastward; negative = westward (m/s).

v10 – 10-meter v-component of wind: north–south wind. Positive = northward; negative = southward (m/s).

d2m – Dew point temperature at 2 meters (K). Indicates moisture in the air.

t2m – Air temperature at 2 meters (K).

msl – Mean sea-level pressure (Pa).

date – Date of the weather observation with time removed (YYYY-MM-DD).

Tornado event variables (from NOAA / SPC dataset)

event_id – Unique ID assigned in the script to identify each tornado report.

yr – Year of the tornado event.

mo – Month of the tornado event.

dy – Day of the tornado event.

st – State abbreviation (e.g., TX).

mag – Tornado magnitude (F-scale rating from 0 to 5).

inj – Number of injuries caused by the tornado.

fat – Number of fatalities caused by the tornado.

slat – Tornado starting latitude (degrees).

slon – Tornado starting longitude (degrees).

elat – Tornado ending latitude (degrees).

elon – Tornado ending longitude (degrees).

len – Tornado path length (miles).

wid – Tornado path width (yards).

dist2 – Squared distance between tornado start location and nearest weather grid point (used to pick the closest match).