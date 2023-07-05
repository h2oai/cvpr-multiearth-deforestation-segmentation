PRECISION = 3
IMAGE_SIZE = 256

LAST_MONTH = 12
LAST_YEAR = 2020


SOUTH_AMERICA = "sa"

REGIONS = [SOUTH_AMERICA]


south_america_coordinates = [[
    [-105, -57],
    [-105, 19],
    [-34, 19],
    [-34, -57],
    [-105, -57]
]]

# NASA input SAR ~ 500m pixel resolution
MAP_SHAPE = {
    SOUTH_AMERICA: (18242, 17041)
}

# Target burned area ~ 250m pixel resolution
FIRECCI_SHAPE = {
    SOUTH_AMERICA: (33842, 31616)
}

COMP_AREA_LATS = (-4.39, -3.33)
COMP_AREA_LONS = (-55.2, -54.48)